import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm_notebook
import datetime
from skimage.segmentation import mark_boundaries
import lime.lime_image
import shap
from shap.plots import colors
import matplotlib
import matplotlib.pyplot as plt


# ハイパパラメータ設定
batch_size = 8
lr = .0001  # for sgd
epochs = 30
log_interval = 1000
num_classes = 256
# GPU利用するなら設定
device = torch.device('cuda')

# 事前学習モデルの利用
model = torchvision.models.resnet50(pretrained=True)

# fine-tuningのために全結合層以外の層の重みを固定
for name, param in model.named_parameters():
    if not name.startswith('fc'):
        param.detach_()

# 出力層をクラス数に合わせる
model.fc = nn.Linear(model.fc.in_features, num_classes)

# GPU利用するなら設定
model.cuda()
model.to(device)

# optimizer設定
optimizer = optim.SGD(model.parameters(), lr=lr)
# loss関数設定
loss_func = nn.CrossEntropyLoss()

# 学習用データセット読み込み
train_dataset = datasets.ImageFolder(
    './train',
    # 画像前処理設定
    transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# 学習実行
model.train()
for epoch in tqdm_notebook(range(1, epochs + 1)):
    for batch_idx, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 学習済みモデル保存
today = datetime.date.today()
path_pytorch_model = 'torch_model/resnet50_' + today.strftime('%Y%m%d') + '.pth'
torch.save(model.state_dict(), path_pytorch_model)

# 作成したモデルの評価
val_dataset = datasets.ImageFolder(
    './validation',
    transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()])
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    shuffle=False
)

# ラベル一覧
list_labels = val_dataset.classes

count_all = 0
count_acc = 0

model.eval()
with torch.no_grad():
    for images_val, target_val in val_loader:
        count_all += 1
        images_val = images_val.to(device)
        target_val = target_val.to(device)
        output_val = model(images_val)
        if output_val[0].argmax().data == target_val.data:
            count_acc += 1

print('Acc: ' + str(count_acc / count_all))

# 説明用画像
path_zebra = './validation/250.zebra'
path_zebra_data = sorted([os.path.join(
    path_zebra, i) for i in os.listdir(path_zebra)])


# 関数定義
# LIME用の予測（推論）関数
def predict_for_lime(images):
    ret_predict_scores = []

    with torch.no_grad():

        for image in images:

            # resize iage
            to_pil_image = transforms.ToPILImage()
            pil_image = to_pil_image(image)
            image = torchvision.transforms.functional.resize(
                pil_image, (224, 224))

            # to Tensor
            to_tensor = torchvision.transforms.ToTensor()
            tensor_image = to_tensor(np.asarray(image))
            tensor_image = tensor_image[None, :]

            tensor_image = tensor_image.to(device)

            # predict
            output_val = model(tensor_image)
            output_val = output_val.cpu()

            ret_predict_scores.append(output_val.numpy()[0])

    return np.array(ret_predict_scores)


# LIMEによる説明
# クラスオブジェクト初期化
explainer = lime.lime_image.LimeImageExplainer(random_state=0)

index_lime = 6

# 画像の読み込み
with open(path_zebra_data[index_lime], 'rb') as f:
    img = Image.open(f)
    img_pil = img.convert('RGB')
    img_pil = np.asarray(img_pil)

explanation = explainer.explain_instance(
    img_pil,
    predict_for_lime,
    hide_color=0.9,
    num_features=100,
    num_samples=1000)  # number of images that will be sent to classification function

# 説明用画像出力
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry = mark_boundaries(temp / 255.0, mask)
plt.imshow(img_boundry)

# SHAPによる説明
# SHAPによる説明に利用する画像の生成
# 画像の変換に用いるメソッド定義
to_pil_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# 画像用変数の初期化
tensor_images = torch.Tensor()

for path_data in path_zebra_data:
    with open(path_data, 'rb') as f:
        img = Image.open(f)
        img_pil = img.convert('RGB')
        img_pil = np.asarray(img_pil)

    tmp_pil_image = to_pil_image(img_pil)
    tmp_resize_image = transforms.functional.resize(
        tmp_pil_image, (224, 224))

    # to Tensor
    tmp_tensor_image = to_tensor(np.asarray(tmp_resize_image))
    tensor_images = torch.cat((tensor_images, tmp_tensor_image[None, :]))

index_shap = 6

# GPU利用するなら設定
tensor_images = tensor_images.to(device)

# SHAPによる説明1
# クラスオブジェクト初期化
grad_explainer1 = shap.GradientExplainer(model, tensor_images)
shap_values1, indexes1 = grad_explainer1.shap_values1(
    tensor_images[index_shap:index_shap+1], ranked_outputs=2, rseed=0, output_rank_order='max')

# 予測ラベル名取得
index_names1 = np.vectorize(lambda x: list_labels[x])(indexes1.cpu())
to_explain = np.array(tensor_images[index_shap:index_shap+1].cpu())

# 画像描画
labels = index_names1
width = 20
aspect = 0.2
hspace = 0.2
labelpad = None
show = True

multi_output = True
if type(shap_values1) != list:
    multi_output = False
    shap_values1 = [shap_values1]

# make sure labels
if labels is not None:
    assert labels.shape[0] == shap_values1[0].shape[0], "Labels must have same row count as shap_values1 arrays!"
    if multi_output:
        assert labels.shape[1] == len(shap_values1), "Labels must have a column for each output in shap_values1!"
    else:
        assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values1."

label_kwargs = {} if labelpad is None else {'pad': labelpad}

# plot our explanations
x = to_explain.transpose(0, 2, 3, 1)
fig_size = np.array([3 * (len(shap_values1) + 1), 2.5 * (x.shape[0] + 1)])

if fig_size[0] > width:
    fig_size *= width / fig_size[0]

fig, axes = plt.subplots(
    nrows=x.shape[0], ncols=len(shap_values1) + 1, figsize=fig_size)

if len(axes.shape) == 1:
    axes = axes.reshape(1, axes.size)

for row in range(x.shape[0]):
    x_curr = x[row].copy()

    # make sure
    if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
        x_curr = x_curr.reshape(x_curr.shape[:2])
    if x_curr.max() > 1:
        x_curr /= 255.

    # get a grayscale version of the image
    if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
        x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
    else:
        x_curr_gray = x_curr

    axes[row, 0].imshow(x_curr, cmap=plt.get_cmap('gray'))
    axes[row, 0].axis('off')

    if len(shap_values1[0][row].shape) == 2:
        abs_vals = np.stack([np.abs(shap_values1[i]) for i in range(len(shap_values1))], 0).flatten()
    else:
        abs_vals = np.stack([np.abs(shap_values1[i].sum(-1)) for i in range(len(shap_values1))], 0).flatten()

    max_val = np.nanpercentile(abs_vals, 99.9)

    for i in range(len(shap_values1)):
        if labels is not None:
            axes[row,i+1].set_title(labels[row, i], **label_kwargs)

        shap_value = shap_values1[i]
        shap_value = shap_value.transpose(0, 2, 3, 1)

        # sv = shap_values1[i][row] if len(shap_values1[i][row].shape) == 2 else shap_values1[i][row].sum(-1)
        sv = shap_value[row] if len(shap_value[row].shape) == 2 else shap_value[row].sum(-1)
        axes[row, i+1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[1], sv.shape[0], -1))
        im = axes[row,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        axes[row, i+1].axis('off')

if hspace == 'auto':
    fig.tight_layout()
else:
    fig.subplots_adjust(hspace=hspace)

cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0]/aspect)

cb.outline.set_visible(False)
if show:
    plt.show()

fig.savefig('shap_gradient1_シマウマ.png')

# SHAPによる説明2
grad_explainer2 = shap.GradientExplainer(
    (model, model.layer2), tensor_images, local_smoothing=0)
shap_values2, indexes2 = grad_explainer2.shap_values(
    tensor_images[index_shap:index_shap+1], ranked_outputs=2, rseed=0, output_rank_order='max')

# 予測ラベル名取得
index_names2 = np.vectorize(lambda x: list_labels[x])(indexes2.cpu())
to_explain = np.array(tensor_images[index_shap:index_shap+1].cpu())

# 画像描画
labels = index_names2
width = 20
aspect = 0.2
hspace = 0.2
labelpad = None
show = True

multi_output = True
if type(shap_values2) != list:
    multi_output = False
    shap_values2 = [shap_values2]

# make sure labels
if labels is not None:
    assert labels.shape[0] == shap_values2[0].shape[0], "Labels must have same row count as shap_values arrays!"
    if multi_output:
        assert labels.shape[1] == len(shap_values2), "Labels must have a column for each output in shap_values!"
    else:
        assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

label_kwargs = {} if labelpad is None else {'pad': labelpad}

# plot our explanations
x = to_explain.transpose(0, 2, 3, 1)
fig_size = np.array([3 * (len(shap_values2) + 1), 2.5 * (x.shape[0] + 1)])

if fig_size[0] > width:
    fig_size *= width / fig_size[0]

fig, axes = plt.subplots(
    nrows=x.shape[0], ncols=len(shap_values2) + 1, figsize=fig_size)

if len(axes.shape) == 1:
    axes = axes.reshape(1, axes.size)

for row in range(x.shape[0]):
    x_curr = x[row].copy()

    # make sure
    if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
        x_curr = x_curr.reshape(x_curr.shape[:2])
    if x_curr.max() > 1:
        x_curr /= 255.

    # get a grayscale version of the image
    if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
        x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
    else:
        x_curr_gray = x_curr

    axes[row, 0].imshow(x_curr, cmap=plt.get_cmap('gray'))
    axes[row, 0].axis('off')

    if len(shap_values2[0][row].shape) == 2:
        abs_vals = np.stack([np.abs(shap_values2[i]) for i in range(len(shap_values2))], 0).flatten()
    else:
        abs_vals = np.stack([np.abs(shap_values2[i].sum(-1)) for i in range(len(shap_values2))], 0).flatten()

    max_val = np.nanpercentile(abs_vals, 99.9)

    for i in range(len(shap_values2)):
        if labels is not None:
            axes[row,i+1].set_title(labels[row, i], **label_kwargs)

        shap_value = shap_values2[i]
        shap_value = shap_value.transpose(0, 2, 3, 1)

        # sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
        sv = shap_value[row] if len(shap_value[row].shape) == 2 else shap_value[row].sum(-1)
        axes[row, i+1].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[1], sv.shape[0], -1))
        im = axes[row,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
        axes[row, i+1].axis('off')

if hspace == 'auto':
    fig.tight_layout()
else:
    fig.subplots_adjust(hspace=hspace)

cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0]/aspect)

cb.outline.set_visible(False)
if show:
    plt.show()

fig.savefig('shap_gradient2_シマウマ.png')