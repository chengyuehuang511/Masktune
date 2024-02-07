from src.models.simple_cnn import SmallCNN
from src.models.resnet50 import ResNet50
from src.models.resnet import resnet32
from src.models.vgg16 import vgg16_bn


# def get_x(self, x):
#     x = F.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
#     return chainer.cuda.to_cpu(x.data)

# def get_top_h(self, x):
#     h = x
#     for i in range(self.num_layers):
#         h = F.relu(getattr(self, 'conv{}_1'.format(i))(h))
#         h = F.relu(getattr(self, 'conv{}_2'.format(i))(h))
#         h = F.max_pooling_2d(h, self.pool_sizes[i])
#     h = F.average(h, axis=(2, 3))
#     return chainer.cuda.to_cpu(h.data)

# def get_all_h(self, x):
#     hs = []
#     h = x
#     # hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
#     for i in range(self.num_layers):
#         h = F.relu(getattr(self, 'conv{}_1'.format(i))(h))
#         hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
#         h = F.relu(getattr(self, 'conv{}_2'.format(i))(h))
#         hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
#         h = F.max_pooling_2d(h, self.pool_sizes[i])
#         hs.append(F.reshape(h, (h.shape[0], h.shape[1] * h.shape[2] * h.shape[3])))
#     h = F.average(h, axis=(2, 3))
#     hs.append(h)
#     ret_h = F.hstack(hs)
#     return chainer.cuda.to_cpu(ret_h.data)