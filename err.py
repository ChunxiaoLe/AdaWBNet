from evaluation.evaluate_cc import evaluate_cc
import numpy as np


def err_compution(gt,out_img):
    gt_ = gt.cpu().detach().numpy().transpose(1,2,0)
    out_img_ =out_img.cpu().detach().numpy().transpose(1,2,0)
    deltaE00, MSE, MAE = evaluate_cc(out_img_ * 255, gt_ * 255, 0, opt=3)
    return deltaE00,MSE,MAE

def err_compution_npy(gt,out_img):
    # gt_ = gt.cpu().detach().numpy().transpose(1,2,0)
    # out_img_ =out_img.cpu().detach().numpy().transpose(1,2,0)
    deltaE00, MSE, MAE = evaluate_cc(out_img * 255, gt * 255, 0, opt=3)
    return deltaE00,MSE,MAE

def error_evaluation(error_list):
    es = np.array(error_list)
    es.sort()
    ae = np.array(es).astype(np.float32)

    x, y, z = np.percentile(ae, [25, 50, 75])
    Mean = np.mean(ae)
    print("Mean\tQ1\tQ2\tQ3")
    print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(Mean, x, y, z))
    return Mean,x,y,z