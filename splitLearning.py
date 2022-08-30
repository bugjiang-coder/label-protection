import math

import torch
from solver import solve_isotropic_covariance, symKL_objective


# 这个类就是增加了一个user_id属性
# 同时增加了一个backward方法，这个方法是用来更新模型参数的
# 与传统的在类的外面backward不同,这里是在类的内部定义了一个方法
class BaseClient(torch.nn.Module):
    """Abstract class foe the client of collaborative learning.
    Args:
        model (torch.nn.Module): a local model
        user_id (int, optional): id of this client. Defaults to 0.
    """

    def __init__(self, model, user_id=0):
        """Initialize BaseClient"""
        super(BaseClient, self).__init__()
        self.model = model
        self.user_id = user_id

    def forward(self, x):
        return self.model(x)

    def upload(self):
        """Upload the locally learned informatino to the server."""
        pass

    def download(self):
        """Download the global model from the server."""
        pass

    def train(self):
        # 将模型设置为训练模式
        self.model.train()

    def eval(self):
        self.model.eval()

    def backward(self, loss):
        """Execute backward mode automatic differentiation with the give loss.
        Args:
            loss (torch.Tensor): the value of calculated loss.
        """
        loss.backward()


# 该类需要一个model参数，这个参数是一个torch.nn.Module的实例
# 这个model用来实现forward方法
class SplitNNClient(BaseClient):
    def __init__(self, model, user_id=0):
        super().__init__(model, user_id=user_id)
        # 增加了作为中间层的一些属性
        # 自己的输出
        self.own_intermidiate = None
        # 前一模型的输入
        self.prev_intermidiate = None
        # 后面模型的梯度
        self.grad_from_next_client = None

    def forward(self, prev_intermediate):
        """Send intermidiate tensor to the server
        Args:
            x (torch.Tensor): the input data
        Returns:
            intermidiate_to_next_client (torch.Tensor): the output of client-side
                                                   model which the client sent
                                                   to the server
        """
        # 保存上一个模型的输入
        self.prev_intermidiate = prev_intermediate
        # 保存前向传播的结果
        self.own_intermidiate = self.model(prev_intermediate)
        # 返回前向传播的结果，该结果的梯度与该模型进行了解绑
        '''
        detach（）就是截断反向传播的梯度流。detach()函数会返回一个新的Tensor对象b，
        并且新Tensor是与当前的计算图分离的
        其requires_grad属性为False，反向传播时不会计算其梯度
        '''
        intermidiate_to_next_client = self.own_intermidiate.detach().requires_grad_()
        return intermidiate_to_next_client

    def upload(self, x):
        # 上传的是自己的forword的结果
        return self.forward(x)

    def download(self, grad_from_next_client):
        self._client_backward(grad_from_next_client)

    def _client_backward(self, grad_from_next_client):
        """Client-side back propagation
        Args:
            grad_from_server: gradient which the server send to the client
        """
        self.grad_from_next_client = grad_from_next_client
        self.own_intermidiate.backward(grad_from_next_client)

    def distribute(self):
        return self.prev_intermidiate.grad.clone()


class SplitNN(torch.nn.Module):
    def __init__(self, clients, optimizers):
        super().__init__()
        self.clients = clients
        self.optimizers = optimizers
        self.num_clients = len(clients)
        self.recent_output = None

    def forward(self, x):
        intermidiate_to_next_client = x
        for client in self.clients:
            intermidiate_to_next_client = client.upload(intermidiate_to_next_client)
        output = intermidiate_to_next_client
        self.recent_output = output
        return output

    def backward(self, loss):
        # 这里是不是有问题？
        # 传入的梯度是output的梯度，但是这里在loss.backward()
        # 可能样例代码要修改
        loss.backward()
        return self.backward_gradient(self.recent_output.grad)

    def backward_gradient(self, grads_outputs):
        # 从后向前传播
        grad_from_next_client = grads_outputs
        for i in range(self.num_clients - 1, -1, -1):
            self.clients[i].download(grad_from_next_client)
            if i != 0:
                grad_from_next_client = self.clients[i].distribute()
        return grad_from_next_client

    def train(self):
        for client in self.clients:
            client.train()

    def eval(self):
        for client in self.clients:
            client.train()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class ISO_SplitNN(torch.nn.Module):
    def __init__(self, clients, optimizers, t=1):
        super().__init__()
        self.clients = clients
        self.optimizers = optimizers
        self.num_clients = len(clients)
        self.recent_output = None
        self.t = t

    def forward(self, x):
        intermidiate_to_next_client = x
        for client in self.clients:
            intermidiate_to_next_client = client.upload(intermidiate_to_next_client)
        output = intermidiate_to_next_client
        self.recent_output = output
        return output

    def backward(self, loss):
        # 这里是不是有问题？
        # 传入的梯度是output的梯度，但是这里在loss.backward()
        # 可能样例代码要修改
        loss.backward()
        return self.backward_gradient(self.recent_output.grad)

    def backward_gradient(self, grads_outputs):
        # 从后向前传播
        grad_from_next_client = grads_outputs
        for i in range(self.num_clients - 1, -1, -1):
            self.clients[i].download(grad_from_next_client)
            if i != 0:
                # 由于只用2个参与者所以噪声直接加在这里
                grad_from_next_client = self.clients[i].distribute() + self.t * torch.randn(grads_outputs.shape)
        return grad_from_next_client

    def train(self):
        for client in self.clients:
            client.train()

    def eval(self):
        for client in self.clients:
            client.train()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class MAX_NORM_SplitNN(torch.nn.Module):
    def __init__(self, clients, optimizers):
        super().__init__()
        self.clients = clients
        self.optimizers = optimizers
        self.num_clients = len(clients)
        self.recent_output = None

    def forward(self, x):
        intermidiate_to_next_client = x
        for client in self.clients:
            intermidiate_to_next_client = client.upload(intermidiate_to_next_client)
        output = intermidiate_to_next_client
        self.recent_output = output
        return output

    def backward(self, loss):
        # 这里是不是有问题？
        # 传入的梯度是output的梯度，但是这里在loss.backward()
        # 可能样例代码要修改
        loss.backward()
        return self.backward_gradient(self.recent_output.grad)

    def backward_gradient(self, grads_outputs):
        # 从后向前传播
        grad_from_next_client = grads_outputs
        for i in range(self.num_clients - 1, -1, -1):
            self.clients[i].download(grad_from_next_client)
            if i != 0:
                # 由于只用2个参与者所以噪声直接加在这里
                g = self.clients[i].distribute()
                # print(g.shape)
                g_norm = g.pow(2).sum(dim=1).sqrt()
                # print(g_norm.shape)
                max_norm = torch.max(g_norm)
                stds = torch.maximum(max_norm ** 2 / (g_norm ** 2 + 1e-32) - 1.0, torch.zeros(size=g_norm.shape)).sqrt()
                standard_gaussian_noise = torch.normal(mean=0.0, std=1.0, size=stds.shape)
                gaussian_noise = standard_gaussian_noise * stds
                # print(gaussian_noise.shape)
                # print((torch.ones(size=stds.shape) + gaussian_noise).shape)
                # print((torch.ones(size=stds.shape) + gaussian_noise).t().shape)
                grad_from_next_client = g * (torch.ones(size=stds.shape) + gaussian_noise).reshape([-1, 1])
        return grad_from_next_client

    def train(self):
        for client in self.clients:
            client.train()

    def eval(self):
        for client in self.clients:
            client.train()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class Marvell_SplitNN(torch.nn.Module):
    def __init__(self, clients, optimizers):
        super().__init__()
        self.clients = clients
        self.optimizers = optimizers
        self.num_clients = len(clients)
        self.recent_output = None

    def forward(self, x):
        intermidiate_to_next_client = x
        for client in self.clients:
            intermidiate_to_next_client = client.upload(intermidiate_to_next_client)
        output = intermidiate_to_next_client
        self.recent_output = output
        return output

    def backward(self, loss, label):
        # 由于Marvell算法需要知道标签，所以必须反向传播的时候需要传入label
        # 这里是不是有问题？
        # 传入的梯度是output的梯度，但是这里在loss.backward()
        # 可能样例代码要修改
        loss.backward()
        return self.backward_gradient(self.recent_output.grad, label)

    def backward_gradient(self, grads_outputs, label):
        # 从后向前传播
        grad_from_next_client = grads_outputs
        for i in range(self.num_clients - 1, -1, -1):
            self.clients[i].download(grad_from_next_client)
            if i != 0:
                # 由于只用2个参与者所以噪声直接加在这里
                g = self.clients[i].distribute()
                grad_from_next_client = marvell_g(g, label)

        return grad_from_next_client

    def train(self):
        for client in self.clients:
            client.train()

    def eval(self):
        for client in self.clients:
            client.train()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


def marvell_g(g, labels):
    # the batch label was stored in shared_var.batch_y in train_and_test

    # 经典改变形态，没有任何用
    g_original_shape = g.shape
    g = torch.reshape(g, shape=(g_original_shape[0], -1))

    # 批次标签存储在 train_and_test 中的 shared_var.batch_y 中
    # 首先要获取标签
    y = labels
    # print(y)
    # print(y.shape)
    # print(y[:,0] == 1)
    pos_g = g[y[:,0] == 1]
    # 按照某个轴，对张量求平均值 沿着第0个轴求平均，相当于对16个维度求平均
    # 这里是把16个隐藏的梯度看作16个正态分布，求这些正态分布的均值。
    pos_g_mean = torch.mean(pos_g, dim=0, keepdim=True)  # shape [1, d]
    # 求出了16个正态分布的方差
    pos_coordinate_var = torch.mean(torch.square(pos_g - pos_g_mean), dim=0)  # use broadcast

    neg_g = g[y[:,0] == 0]
    # 16个正态分布的均值
    neg_g_mean = torch.mean(neg_g, dim=0, keepdim=True)  # shape [1, d]
    # 16个正态分布的方差
    neg_coordinate_var = torch.mean(torch.square(neg_g - neg_g_mean), dim=0)

    # 求两种标签方差的均值：
    avg_pos_coordinate_var = torch.mean(pos_coordinate_var)
    avg_neg_coordinate_var = torch.mean(neg_coordinate_var)
    # print('pos', avg_pos_coordinate_var)
    # print('neg', avg_neg_coordinate_var)

    # 比较16个维度均值上的差异
    g_diff = pos_g_mean - neg_g_mean
    # print(pos_g_mean)
    # print(neg_g_mean)

    # 算出16个差异的模
    g_diff_norm = float(torch.norm(g_diff).numpy())
    # print(g_diff_norm)
    if g_diff_norm ** 2 > 1:
        print('pos_g_mean', pos_g_mean.shape)
        print('neg_g_mean', neg_g_mean.shape)
        assert g_diff_norm

    # 将u v设置为方差的均值 v为正的 u为负的
    u = float(avg_neg_coordinate_var)
    v = float(avg_pos_coordinate_var)
    # if u == 0.0:
    #     print('neg_g')
    #     print(neg_g)
    # if v == 0.0:
    #     print('pos_g')
    #     print(pos_g)

    # d表示有一个批次有几个数据
    d = float(g.shape[1])

    # 按一定方式计算张量中元素之和  获得flag为1的比例
    p = float(torch.sum(y) / len(y))# p is set as the fraction of positive in the batch

    # print('u={0},v={1},d={2},g={3},p={4},P={5}'.format(u,v,d,g_diff_norm**2,p,P))

    # 默认为1.0 是不是唯一的可调参数？ init_scale
    scale = 1.0
    # scale为1表示 限制 正负两种噪声的强度小于正负两种梯度之差

    # 开始迭代优化

    lam10, lam20, lam11, lam21 = None, None, None, None
    while True:
        P = scale * g_diff_norm ** 2
        # print('g_diff_norm ** 2', g_diff_norm ** 2)
        # print('P', P)
        # print('u, v, d, p, g_diff\n', u, v, d, p, g_diff)
        lam10, lam20, lam11, lam21, sumKL = \
            solve_isotropic_covariance(
                u=u,
                v=v,
                d=d,
                g=g_diff_norm ** 2,
                p=p,
                P=P,
                lam10_init=lam10,
                lam20_init=lam20,
                lam11_init=lam11,
                lam21_init=lam21)
        # print('sumKL', sumKL)
        # print()

        # print(scale)
        # if not dynamic or sumKL <= sumKL_threshold:
        break

        # scale *= 1.5  # loosen the power constraint
    if sumKL == -1:
    #     这里为了解决bug被迫这样写
        return torch.reshape(g, shape=g_original_shape)
    perturbed_g = g
    y_float = y.float()

    # positive examples add noise in g1 - g0
    perturbed_g += torch.reshape(torch.multiply(torch.randn(y.shape),
                                                y_float), shape=(-1, 1)) * g_diff * (
                               math.sqrt(lam11 - lam21) / g_diff_norm)

    # add spherical noise to positive examples
    if lam21 > 0.0:
        perturbed_g += torch.randn(g.shape) * torch.reshape(y_float, shape=(-1, 1)) * math.sqrt(lam21)

    # negative examples add noise in g1 - g0
    perturbed_g += torch.reshape(torch.multiply(torch.randn(y.shape),
                                          1 - y_float), shape=(-1, 1)) * g_diff * (
                           math.sqrt(lam10 - lam20) / g_diff_norm)

    # add spherical noise to negative examples
    if lam20 > 0.0:
        perturbed_g += torch.randn(g.shape) * torch.reshape(1 - y_float, shape=(-1, 1)) * math.sqrt(lam20)

    return torch.reshape(perturbed_g, shape=g_original_shape)
