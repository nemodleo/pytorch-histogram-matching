import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_histogram_matching import Histogram_Matching

def test_histogram_matching():
    try:
        # pre
        dst = Image.open('src/dst.jpg').convert('RGB')
        ref = Image.open('src/ref.jpg').convert('RGB')
        dst = np.array(dst)
        ref = np.array(ref)
        dst = transforms.ToTensor()(dst)[None]
        ref = transforms.ToTensor()(ref)[None]
        dst.requires_grad = True
        ref.requires_grad = True
        # gpu
        dst = dst.cuda()
        ref = ref.cuda()
        # histogram matching
        HM = Histogram_Matching(differentiable=True)
        rst = HM(dst, ref)
        assert rst.requires_grad == True, f"rst.requires_grad:{rst.requires_grad == True} != True"
        rst = transforms.ToPILImage()(rst[0])
        rst.save('src/rst.jpg')
        # print(rst*255)
    except Exception as e:
        print(f"[!] Histogram_Matching has a problem:\n\n {e}")

if __name__ == '__main__':
    test_histogram_matching()
