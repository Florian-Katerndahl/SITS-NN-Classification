import torch
import torch.multiprocessing as mp
from time import time

if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    X, Y, L, B = 500, 500, 329, 11

    model = torch.load("lqnld.pkl", map_location=torch.device("cuda")).eval()

    with torch.inference_mode():
        dc = torch.randn(X,Y,L,B)
        split_dc = [torch.squeeze(i, dim=0) for i in torch.vsplit(dc, X)]
        # ds = torch.utils.data.TensorDataset(*split_dc)  # TensorDataset splits along first dimension of input
        # dl = torch.utils.data.DataLoader(ds, batch_size=1024, pin_memory=True, num_workers=4, persistent_workers=True)
        # len(dl) = Y / batch_size; number of batches with at most `batch_size` elements; i.e. first dimension of split_dc divided by `batch_size`
        # each batch in dl contains len(split_dc) elements; we iterate over all X's and predict at most `batch_size` elements
        # idx reaches up to int(Y / batch_size), jdx reaches up to X
        # at [idx, jdx] at most `batch_size` elements are predicted
        # storing in a 1D-array and reshaping afterwards might be the easiest, no? => more applicable to approach below!
        # This approach takes approx. 973 seconds for 500*500 pixels on Gromit without top-level torch.inference_mode() CM 
        # This approach takes approx. 968 seconds for 500*500 pixels on Gromit **with** top-level torch.inference_mode() CM 

        # Alternatively, concatenating split_dc might result in better resource utilization because we're not bound by Y but X * Y
        # This approach takes approx. 1050 seconds for 500*500 pixels on Gromit without top-level torch.inference_mode() CM 
        # This approach takes approx. 1041 seconds for 500*500 pixels on Gromit **with** top-level torch.inference_mode() CM
        ds = torch.utils.data.TensorDataset(torch.cat(split_dc, 0))  # TensorDataset splits along first dimension of input
        dl = torch.utils.data.DataLoader(ds, batch_size=1024, pin_memory=True, num_workers=4, persistent_workers=True)
        
        t0 = time()
        for idx, batch in enumerate(dl):
            for jdx, samples in enumerate(batch):
                print(len(samples))
                output = model(samples.cuda(non_blocking=True))
                _, pred = torch.max(output, dim=1)
                # with torch.no_grad():
                #     output = model(samples.cuda(non_blocking=True))
                #     _, pred = torch.max(output, dim=1)
        
        print("Prediction of chunk took", time() - t0, "seconds")
