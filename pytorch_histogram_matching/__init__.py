import torch
import torch.nn as nn
import torch.nn.functional as F

class Histogram_Matching(nn.Module):
    def __init__(self):
        super(Histogram_Matching, self).__init__()   

    def forward(self, dst, ref):
        # B C
        B, C, H, W = dst.size()
        # assertion
        assert H == W
        assert dst.device == ref.device
        # [B C H W]
        dst = dst.detach().clone()
        ref = ref.detach().clone()
        # [B C H W]
        dst *= 255
        ref *= 255
        # [B*C 256]
        hist_dst = self.cal_hist(dst)
        hist_ref = self.cal_hist(ref)
        # [B*C 256]
        tables = self.cal_trans(hist_dst, hist_ref)
        # [B C H W]
        rst = dst.detach().clone()
        for b in range(B):
            for c in range(C):
                rst[b,c] = tables[b*c, dst[b,c].long()]
        # [B C H W]
        rst /= 255.
        return rst
        
    def cal_hist(self, img):
        B, C, _, _ = img.size()
        # [B*C 256]
        hists = torch.stack([torch.histc(img[b,c], bins=256, min=0, max=255) for b in range(B) for c in range(C)])
        hists = hists.float()
        hists = F.normalize(hists, p=1)
        # BC 256
        bc, n = hists.size()
        # [B*C 256 256]
        triu = torch.ones(bc, n, n, device=hists.device).triu()
        # [B*C 256]
        hists = torch.bmm(hists[:,None,:], triu)[:,0,:]
        return hists
    
    def cal_trans(self, hist_dst, hist_ref):
        # [B*C 256]
        table = torch.arange(256, device=hist_dst.device).repeat(len(hist_dst), 1)
        for bc in range(len(table)):
            for i in list(range(1, 256)):
                for j in list(range(1, 256)):
                    if hist_dst[bc, i] >= hist_ref[bc, j - 1] and hist_dst[bc, i] <= hist_ref[bc, j]:
                        table[bc, i] = j
                        break
        table[:, 255] = 255
        return table
