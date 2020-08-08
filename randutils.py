import numpy as np
from skimage.io import imshow
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np

def generate_random_circles(n = 100, d = 256):
    circles = np.random.randint(0, d, (n, 3))
    x = np.zeros((d, d), dtype=int)
    f = lambda x, y: ((x - x0)**2 + (y - y0)**2) <= (r/d*10)**2
    for x0, y0, r in circles:
        x += np.fromfunction(f, x.shape)
    x = np.clip(x, 0, 1)

    return x

zf = ZipFile('/content/drive/My Drive/EVA4/tsai/S15EVA4/depth_mask_gt_1k2k.zip')
uncompress_size = sum((file.file_size for file in zf.infolist()))
print('uncompressed_size',uncompress_size/1e6)


def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf
magnet:?xt=urn:btih:63755193F1BD7023B2D2E1BE0DF7BA137713BD89&dn=%20Fan%20Favorite:%20Riley%20Reid%20(Adam%20&%20Eve)%20[2017%20WEB-DL]%20&tr=udp://tracker.coppersurfer.tk:6969/announce&tr=udp://9.rarbg.to:2920/announce&tr=udp://tracker.opentrackr.org:1337&tr=udp://tracker.internetwarriors.net:1337/announce&tr=udp://tracker.leechers-paradise.org:6969/announce&tr=udp://tracker.coppersurfer.tk:6969/announce&tr=udp://tracker.pirateparty.gr:6969/announce&tr=udp://tracker.cyberia.is:6969/announce
    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w

y = generate_random_circles()

wc = {
    0: 1, # background
    1: 5  # objects
}

w = unet_weight_map(y, wc)

imshow(w)


def extract_data_files_new(csv_file, root_dir, start,end, dest_dir="/content/S15EVA4"):
    depthmask_csv = pd.read_csv(csv_file).loc[start:end,:]
    image_file_zip_dict = {val:ZipFile(os.path.join(root_dir,val)) 
                                for val in depthmask_csv['BaseImageFName'].unique()}
    depth_zip_dict = {val:ZipFile(os.path.join(root_dir,val)) 
                                for val in depthmask_csv['DepthImageFName'].unique()}
    baseimage_groups = depthmask_csv.groupby('BaseImageFName')
    depthimage_groups = depthmask_csv.groupby('DepthImageFName')
    bg_image_zip_dict = ZipFile("/content/drive/My Drive/EVA4/tsai/S15EVA4/FinalDataSet/bg_images.zip")
    
    #pbar = tqdm(range(len(depthmask_csv)))
    print("Extracting image and mask files")
    for zip_name in list(baseimage_groups.groups.keys()):
        zip_obj = ZipFile(os.path.join(root_dir,zip_name))
        img_mask_arr = np.hstack(depthmask_csv.loc[baseimage_groups.groups[zip_name],['ImageName','MaskName']].values)
        pbar = tqdm(img_mask_arr)
        pbar.set_description(zip_name)
        for batch_idx, image_name in enumerate(pbar):
            zip_obj.extract(image_name)
    
    for zip_name in list(depthimage_groups.groups.keys()):
        zip_obj = ZipFile(os.path.join(root_dir,zip_name))
        img_mask_arr = np.hstack(depthmask_csv.loc[depthimage_groups.groups[zip_name],['Depthname']].values)
        pbar = tqdm(img_mask_arr)
        pbar.set_description(zip_name)
        for batch_idx, image_name in enumerate(pbar):
            zip_obj.extract(image_name)

    for depth_image_name in depthmask_csv['BGImageName'].unique():
        bg_image_zip_dict.extract(depth_image_name)


    #for idx in range(len(depthmask_csv)):
    # for batch_idx,idx in enumerate(pbar):
    #     base_img_name = depthmask_csv.iloc[idx, 0]
    #     mask_img_name = depthmask_csv.iloc[idx, 1]
    #     depth_img_name = depthmask_csv.iloc[idx, 2]
    #     bg_image_name = depthmask_csv.iloc[idx, 3]
    #     base_img_zip = image_file_zip_dict[depthmask_csv.iloc[idx, 4]]
    #     depth_img_zip = depth_zip_dict[depthmask_csv.iloc[idx, 5]]
    #     #self.depthmask_csv = pd.read_csv(csv_file)
        
    #     base_img = base_img_zip.extract(base_img_name, dest_dir)
    #     bg_img = bg_image_zip_dict.extract(bg_image_name, dest_dir)

    #     ### GT labels 
    #     mask_img = base_img_zip.extract(mask_img_name, dest_dir)
    #     depth_img = depth_img_zip.extract(depth_img_name, dest_dir)
    print("Total file count{} ".format(len(glob.glob(dest_dir+"/*jpg"))))
                                                    

class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p:nn.Parameter): self.val = p
    def forward(self, x): return x

def children_and_parameters(m:nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children
def children(m:nn.Module)->nn.ModuleList:
    "Get children of `m`."
    return list(m.children())

def num_children(m:nn.Module)->int:
    "Get number of children modules in `m`."
    return len(children(m))

flatten_model = lambda m: sum(map(flatten_model,children_and_parameters(m)),[]) if num_children(m) else [m]
def freeze_to(self, n:int)->None:
    "Freeze layers up to layer group `n`."
    if hasattr(self.model, 'reset'): self.model.reset()
    for g in self.layer_groups[:n]:
        for l in g:
            if not self.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
    for g in self.layer_groups[n:]: requires_grad(g, True)
    self.create_opt(defaults.lr)


def torch_grid_create()
    for class_idx in range(len(classes_abv)):
        fig = plt.figure(figsize=(100,100))
        image_list = [erro_img[idx] for idx in np.where(actuals == class_idx)[0][:10]]
        #for idx,pos in enumerate(np.where(actuals == class_idx)[0][:10]):
        grid = make_grid(image_list, nrow=5, padding=0, normalize=True, pad_value=0)
        npgrid = grid.cpu().numpy()
        ax = fig.add_subplot(10, 2, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    #    plt.imshow(np.transpose(error_images[pos].cpu().numpy(), (1, 2, 0)))
        #ax.set(ylabel="Pred="+classes[np.int(preds[pos])], xlabel="Actual="+classes[np.int(actuals[pos])])