import torch
import torch.nn as nn
import torch.nn.init as init
from math import sqrt
import math
import torch.nn.functional as F

def self_attention(query, key, value, dropout=None, mask=None):
    """
    自注意力计算
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: drop比率
    :param mask: 是否mask
    :return: 经自注意力机制计算后的值
    """
    d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
    # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    # if mask is not None:
    #     """
    #     scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
    #     在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
    #     """
    #     # mask.cuda()
    #     # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引
    #
    #   scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # 判断是否要对相似概率分布进行dropout操作
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return torch.matmul(self_attn_softmax, value), self_attn_softmax

def multi_head_attention(query, key, value, head, d_model,dropout=None, mask=None):
    d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
    # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax

    return torch.matmul(self_attn_softmax, value), self_attn_softmax

class MultiHeadAttention(nn.Module):
    """
    多头注意力计算
    """

    def __init__(self, head, d_model, dropout=0.1):
        """
        :param head: 头数
        :param d_model: 词向量的维度，必须是head的整数倍
        :param dropout: drop比率
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)  # 确保词向量维度是头数的整数倍
        self.d_k = d_model // head  # 被拆分为多头后的某一头词向量的维度
        self.head = head
        self.d_model = d_model

        """
        由于多头注意力机制是针对多组Q、K、V，因此有了下面这四行代码，具体作用是，
        针对未来每一次输入的Q、K、V，都给予参数进行构建
        其中linear_out是针对多头汇总时给予的参数
        """
        self.linear_query = nn.Linear(d_model, d_model)  # 进行一个普通的全连接层变化，但不修改维度
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None  # attn_softmax是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(QK^T)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            """
            多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第二维（head维）添加一维，与后面的self_attention计算维度一样
            具体点将，就是：
            因为mask的作用是未来传入self_attention这个函数的时候，作为masked_fill需要mask哪些信息的依据
            针对多head的数据，Q、K、V的形状维度中，只有head是通过view计算出来的，是多余的，为了保证mask和
            view变换之后的Q、K、V的形状一直，mask就得在head这个维度添加一个维度出来，进而做到对正确信息的mask
            """
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)  # batch_size大小，假设query的维度是：[10, 32, 512]，其中10是batch_size的大小

        """
        下列三行代码都在做类似的事情，对Q、K、V三个矩阵做处理
        其中view函数是对Linear层的输出做一个形状的重构，其中-1是自适应（自主计算）
        从这种重构中，可以看出，虽然增加了头数，但是数据的总维度是没有变化的，也就是说多头是对数据内部进行了一次拆分
        transopose(1,2)是对前形状的两个维度(索引从0开始)做一个交换，例如(2,3,4,5)会变成(2,4,3,5)
        因此通过transpose可以让view的第二维度参数变成n_head
        假设Linear成的输出维度是：[10, 32, 512]，其中10是batch_size的大小
        注：这里解释了为什么d_model // head == d_k，如若不是，则view函数做形状重构的时候会出现异常
        """
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]，head=8
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        # x是通过自注意力机制计算出来的值， self.attn_softmax是相似概率分布
        x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask)

        """
        下面的代码是汇总各个头的信息，拼接后形成一个新的x
        其中self.head * self.d_k，可以看出x的形状是按照head数拼接成了一个大矩阵，然后输入到linear_out层添加参数
        contiguous()是重新开辟一块内存后存储x，然后才可以使用.view方法，否则直接使用.view方法会报错
        """
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)

# Weight & Bias Initialization
def initialization(net):
    if isinstance(net, nn.Linear):
        init.xavier_uniform(net.weight)
        init.zeros_(net.bias)

# SDGCCA with 3 Modality
class SDGCCA_3_M(nn.Module):
    def __init__(self, m1_embedding_list, m2_embedding_list, m3_embedding_list, top_k):
        super(SDGCCA_3_M, self).__init__()
        """
        m1_embedding_list = [d_1, ..., \bar{d_1}]
        m1_embedding_list = [d_2, ..., \bar{d_2}]
        m1_embedding_list = [d_3, ..., \bar{d_3}]
        top_k = k
        """

        # Embedding List of each modality
        m1_du0, m1_du1, m1_du2, m1_du3 = m1_embedding_list
        m2_du0, m2_du1, m2_du2, m2_du3 = m2_embedding_list
        m3_du0, m3_du1, m3_du2 = m3_embedding_list

        # Deep neural network of each modality
        self.model1 = nn.Sequential(
            nn.Linear(m1_du0, m1_du1), nn.Tanh(),
            nn.Linear(m1_du1, m1_du2), nn.Tanh(),
            nn.Linear(m1_du2, m1_du3), nn.Tanh())

        self.model2 = nn.Sequential(
            nn.Linear(m2_du0, m2_du1), nn.Tanh(),
            nn.Linear(m2_du1, m2_du2), nn.Tanh(),
            nn.Linear(m2_du2, m2_du3), nn.Tanh())

        self.model3 = nn.Sequential(
            nn.Linear(m3_du0, m3_du1), nn.Tanh(),
            nn.Linear(m3_du1, m3_du2), nn.Tanh())

        # Weight & Bias Initialization
        self.model1.apply(initialization)
        self.model2.apply(initialization)
        self.model3.apply(initialization)

        self.top_k = top_k

        # Projection matrix
        self.U = None

        # Softmax Function
        self.softmax = nn.Softmax(dim=1)

    # Input: Each modality
    # Output: Deep neural network output of the each modality = [H_1, H_2, H_3]
    def forward(self, x1, x2, x3):
        output1o = self.model1(x1)
        output2o = self.model2(x2)
        output3o = self.model3(x3)
        output1, _ = self_attention(output1o, output1o, output1o, dropout=None, mask=None)
        output2, _ = self_attention(output2o, output2o, output2o, dropout=None, mask=None)
        output3, _ = self_attention(output3o, output3o, output3o, dropout=None, mask=None)
        # Q1 = self.q1(output1o)  # Q: batch_size * seq_len * dim_k
        # K1 = self.k1(output1o)  # K: batch_size * seq_len * dim_k
        # V1 = self.v1(output1o)  # V: batch_size * seq_len * dim_v
        #
        # atten = nn.Softmax(dim=-1)(
        #     torch.bmm(Q1, K1.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        #
        # output1 = torch.bmm(atten, V1)  # Q * K.T() * V # batch_size * seq_len * dim_v
        #
        # Q2 = self.q2(output2o)  # Q: batch_size * seq_len * dim_k
        # K2 = self.k2(output2o)  # K: batch_size * seq_len * dim_k
        # V2 = self.v2(output2o)  # V: batch_size * seq_len * dim_v
        #
        # atten = nn.Softmax(dim=-1)(
        #     torch.bmm(Q2, K2.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        #
        # output2 = torch.bmm(atten, V2)  # Q * K.T() * V # batch_size * seq_len * dim_v
        #
        # Q3 = self.q3(output3o)  # Q: batch_size * seq_len * dim_k
        # K3 = self.k3(output3o)  # K: batch_size * seq_len * dim_k
        # V3 = self.v3(output3o)  # V: batch_size * seq_len * dim_v
        #
        # atten = nn.Softmax(dim=-1)(
        #     torch.bmm(Q3, K3.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        #
        # output3 = torch.bmm(atten, V3)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output1, output2, output3, output1o, output2o, output3o

    # Calculate correlation loss
    def cal_loss(self, H_list, train=True):
        eps = 1e-8
        AT_list = []

        for H in H_list:
            assert torch.isnan(H).sum().item() == 0
            m = H.size(1)  # out_dim
            Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
            assert torch.isnan(Hbar).sum().item() == 0

            A, S, B = Hbar.svd(some=True, compute_uv=True)
            A = A[:, :self.top_k]
            assert torch.isnan(A).sum().item() == 0

            S_thin = S[:self.top_k]
            S2_inv = 1. / (torch.mul(S_thin, S_thin) + eps)
            assert torch.isnan(S2_inv).sum().item() == 0

            T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)
            assert torch.isnan(T2).sum().item() == 0

            T2 = torch.where(T2 > eps, T2, (torch.ones(T2.shape) * eps).to(H.device))
            T = torch.diag(torch.sqrt(T2))
            assert torch.isnan(T).sum().item() == 0

            T_unnorm = torch.diag(S_thin + eps)
            assert torch.isnan(T_unnorm).sum().item() == 0

            AT = torch.mm(A, T)
            AT_list.append(AT)

        M_tilde = torch.cat(AT_list, dim=1)
        assert torch.isnan(M_tilde).sum().item() == 0

        Q, R = M_tilde.qr()
        assert torch.isnan(R).sum().item() == 0
        assert torch.isnan(Q).sum().item() == 0

        U, lbda, _ = R.svd(some=False, compute_uv=True)
        assert torch.isnan(U).sum().item() == 0
        assert torch.isnan(lbda).sum().item() == 0

        G = Q.mm(U[:, :self.top_k])
        assert torch.isnan(G).sum().item() == 0

        U = []  # Projection Matrix

        # Get mapping to shared space
        views = H_list
        F = [H.shape[0] for H in H_list]  # features per view
        for idx, (f, view) in enumerate(zip(F, views)):
            _, R = torch.qr(view)
            Cjj_inv = torch.inverse((R.T.mm(R) + eps * torch.eye(view.shape[1], device=view.device)))
            assert torch.isnan(Cjj_inv).sum().item() == 0
            pinv = Cjj_inv.mm(view.T)

            U.append(pinv.mm(G))

        # If model training -> Change projection matrix
        # Else -> Using projection matrix for calculate correlation loss
        if train:
            self.U = U
            for i in range(len(self.U)):
                self.U[i] = nn.Parameter(torch.tensor(self.U[i]))
        _, S, _ = M_tilde.svd(some=True)

        assert torch.isnan(S).sum().item() == 0
        use_all_singular_values = False
        if not use_all_singular_values:
            S = S.topk(self.top_k)[0]
        corr = torch.sum(S)
        assert torch.isnan(corr).item() == 0

        loss = - corr
        return loss

    # SDGCCA prediction
    # Input: Each modality
    # Output: Soft voting of the label presentation of each modality
    def predict(self, x1, x2, x3):
        # out1 = self.model1(x1)
        # out2 = self.model2(x2)
        # out3 = self.model3(x3)
        output1o = self.model1(x1)
        output2o = self.model2(x2)
        output3o = self.model3(x3)

        out1, _ = self_attention(output1o, output1o, output1o, dropout=None, mask=None)
        out2, _ = self_attention(output2o, output2o, output2o, dropout=None, mask=None)
        out3, _ = self_attention(output3o, output3o, output3o, dropout=None, mask=None)

        t1 = torch.matmul(out1, self.U[0])
        t2 = torch.matmul(out2, self.U[1])
        t3 = torch.matmul(out3, self.U[2])

        y_hat1 = torch.matmul(t1, torch.pinverse(self.U[3]))
        y_hat2 = torch.matmul(t2, torch.pinverse(self.U[3]))
        y_hat3 = torch.matmul(t3, torch.pinverse(self.U[3]))
        y_ensemble = (y_hat1+y_hat2+y_hat3)/3

        y_hat1 = self.softmax(y_hat1)
        y_hat2 = self.softmax(y_hat2)
        y_hat3 = self.softmax(y_hat3)
        y_ensemble = self.softmax(y_ensemble)

        return y_hat1, y_hat2, y_hat3, y_ensemble

class SDGCCA_4_M(nn.Module):
    def __init__(self, m1_embedding_list, m2_embedding_list, m3_embedding_list, m4_embedding_list, top_k):
        super(SDGCCA_4_M, self).__init__()
        """
        m1_embedding_list = [d_1, ..., \bar{d_1}]
        m1_embedding_list = [d_2, ..., \bar{d_2}]
        m1_embedding_list = [d_3, ..., \bar{d_3}]
        top_k = k
        """

        # Embedding List of each modality
        m1_du0, m1_du1, m1_du2, m1_du3 = m1_embedding_list
        m2_du0, m2_du1, m2_du2, m2_du3 = m2_embedding_list
        m3_du0, m3_du1, m3_du2 = m3_embedding_list
        m4_du0, m4_du1, m4_du2, m4_du3 = m4_embedding_list

        # Deep neural network of each modality
        self.model1 = nn.Sequential(
            nn.Linear(m1_du0, m1_du1), nn.Tanh(),
            nn.Linear(m1_du1, m1_du2), nn.Tanh(),
            nn.Linear(m1_du2, m1_du3), nn.Tanh())

        self.model2 = nn.Sequential(
            nn.Linear(m2_du0, m2_du1), nn.Tanh(),
            nn.Linear(m2_du1, m2_du2), nn.Tanh(),
            nn.Linear(m2_du2, m2_du3), nn.Tanh())

        self.model3 = nn.Sequential(
            nn.Linear(m3_du0, m3_du1), nn.Tanh(),
            nn.Linear(m3_du1, m3_du2), nn.Tanh())

        self.model4 = nn.Sequential(
            nn.Linear(m4_du0, m4_du1), nn.Tanh(),
            nn.Linear(m4_du1, m4_du2), nn.Tanh(),
            nn.Linear(m4_du2, m4_du3), nn.Tanh())


        # Weight & Bias Initialization
        self.model1.apply(initialization)
        self.model2.apply(initialization)
        self.model3.apply(initialization)
        self.model4.apply(initialization)

        self.top_k = top_k

        # Projection matrix
        self.U = None

        # Softmax Function
        self.softmax = nn.Softmax(dim=1)

    # Input: Each modality
    # Output: Deep neural network output of the each modality = [H_1, H_2, H_3]
    def forward(self, x1, x2, x3, x4):
        output1o = self.model1(x1)
        output2o = self.model2(x2)
        output3o = self.model3(x3)
        output4o = self.model4(x4)
        output1, _ = self_attention(output1o, output1o, output1o, dropout=None, mask=None)
        output2, _ = self_attention(output2o, output2o, output2o, dropout=None, mask=None)
        output3, _ = self_attention(output3o, output3o, output3o, dropout=None, mask=None)
        output4, _ = self_attention(output4o, output4o, output4o, dropout=None, mask=None)

        return output1, output2, output3, output4, output1o, output2o, output3o, output4o

    # Calculate correlation loss
    # 对于每个视角，做SVD降维，并将得到的A*T矩阵存入列表
    def cal_loss(self, H_list, train=True):
        eps = 1e-8
        AT_list = []

        for H in H_list:
            assert torch.isnan(H).sum().item() == 0
            m = H.size(1)  # out_dim
            Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
            assert torch.isnan(Hbar).sum().item() == 0

            A, S, B = Hbar.svd(some=True, compute_uv=True)
            A = A[:, :self.top_k]
            assert torch.isnan(A).sum().item() == 0

            S_thin = S[:self.top_k]
            S2_inv = 1. / (torch.mul(S_thin, S_thin) + eps)
            assert torch.isnan(S2_inv).sum().item() == 0

            T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)
            assert torch.isnan(T2).sum().item() == 0

            T2 = torch.where(T2 > eps, T2, (torch.ones(T2.shape) * eps).to(H.device))
            T = torch.diag(torch.sqrt(T2))
            assert torch.isnan(T).sum().item() == 0

            T_unnorm = torch.diag(S_thin + eps)
            assert torch.isnan(T_unnorm).sum().item() == 0

            AT = torch.mm(A, T)
            AT_list.append(AT)

        # 将所有视角得到的A*T矩阵拼接成一个大矩阵
        M_tilde = torch.cat(AT_list, dim=1)
        assert torch.isnan(M_tilde).sum().item() == 0

        # 对拼接得到的矩阵做QR分解，并做SVD降维，得到G矩阵
        Q, R = M_tilde.qr()
        assert torch.isnan(R).sum().item() == 0
        assert torch.isnan(Q).sum().item() == 0

        U, lbda, _ = R.svd(some=False, compute_uv=True)
        assert torch.isnan(U).sum().item() == 0
        assert torch.isnan(lbda).sum().item() == 0

        G = Q.mm(U[:, :self.top_k])
        assert torch.isnan(G).sum().item() == 0

        U = []  # Projection Matrix

        # Get mapping to shared space
        # 得到映射矩阵U
        views = H_list
        F = [H.shape[0] for H in H_list]  # features per view
        for idx, (f, view) in enumerate(zip(F, views)):
            _, R = torch.qr(view)
            Cjj_inv = torch.inverse((R.T.mm(R) + eps * torch.eye(view.shape[1], device=view.device)))
            assert torch.isnan(Cjj_inv).sum().item() == 0
            pinv = Cjj_inv.mm(view.T)

            U.append(pinv.mm(G))

        # If model training -> Change projection matrix
        # Else -> Using projection matrix for calculate correlation loss
        if train:
            self.U = U
            for i in range(len(self.U)):
                self.U[i] = nn.Parameter(torch.tensor(self.U[i]))
        _, S, _ = M_tilde.svd(some=True)

        assert torch.isnan(S).sum().item() == 0
        use_all_singular_values = False
        if not use_all_singular_values:
            S = S.topk(self.top_k)[0]
        corr = torch.sum(S)
        assert torch.isnan(corr).item() == 0

        loss = - corr
        return loss

    # SDGCCA prediction
    # Input: Each modality
    # Output: Soft voting of the label presentation of each modality
    def predict(self, x1, x2, x3, x4):
        # out1 = self.model1(x1)
        # out2 = self.model2(x2)
        # out3 = self.model3(x3)
        output1o = self.model1(x1)
        output2o = self.model2(x2)
        output3o = self.model3(x3)
        output4o = self.model4(x4)

        out1, _ = self_attention(output1o, output1o, output1o, dropout=None, mask=None)
        out2, _ = self_attention(output2o, output2o, output2o, dropout=None, mask=None)
        out3, _ = self_attention(output3o, output3o, output3o, dropout=None, mask=None)
        out4, _ = self_attention(output4o, output4o, output4o, dropout=None, mask=None)

        t1 = torch.matmul(out1, self.U[0])
        t2 = torch.matmul(out2, self.U[1])
        t3 = torch.matmul(out3, self.U[2])
        t4 = torch.matmul(out4, self.U[3])

        y_hat1 = torch.matmul(t1, torch.pinverse(self.U[4]))
        y_hat2 = torch.matmul(t2, torch.pinverse(self.U[4]))
        y_hat3 = torch.matmul(t3, torch.pinverse(self.U[4]))
        y_hat4 = torch.matmul(t4, torch.pinverse(self.U[4]))
        y_ensemble = (y_hat1 + y_hat2 + y_hat3 + y_hat4) / 4

        y_hat1 = self.softmax(y_hat1)
        y_hat2 = self.softmax(y_hat2)
        y_hat3 = self.softmax(y_hat3)
        y_hat4 = self.softmax(y_hat4)
        y_ensemble = self.softmax(y_ensemble)

        return y_hat1, y_hat2, y_hat3, y_hat4, y_ensemble
