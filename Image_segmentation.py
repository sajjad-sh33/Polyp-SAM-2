import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from segment_anything import sam_model_registry, SamPredictor

from utils.data_utils import kvasir, CVC_300, CVC_ClinicDB, CVC_ColonDB, etis



    
def setup(args):
    # Prepare model
    if args.model_type == "SAM2":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)
    elif args.model_type == "SAM":
        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
  
    return predictor
    

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def intersectionAndUnion(imPred, imLab, numClass):

	# imPred = imPred * (imLab>0)

	# Compute area intersection:
	intersection = imPred * (imPred==imLab)
	(area_intersection,_) = np.histogram(intersection, bins=numClass, range=(1, numClass))

	# Compute area union:
	(area_pred,_) = np.histogram(imPred, bins=numClass, range=(1, numClass))
	(area_lab,_) = np.histogram(imLab, bins=numClass, range=(1, numClass))
	area_union = area_pred + area_lab - area_intersection
	area_sum = area_pred + area_lab
    
	return (area_intersection, area_union, area_sum)




def Test(args, predictor, dataset):

    arr = dataset.__len__()
    numClass = 1
    area_intersection = np.zeros((numClass, arr))
    area_union = np.zeros((numClass, arr))
    area_sum = np.zeros((numClass, arr))
    
    for i in range(arr):
        
        predictor.set_image(dataset[i]["image"])
        
        if args.model_type == "SAM2":
            if args.using_BBox:
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=dataset[i]["bounding_box"],
                    multimask_output=False,
                    )
            else:
                masks, scores, logits = predictor.predict(
                point_coords=dataset[i]["point_coords"],
                point_labels=dataset[i]["point_labels"],
                multimask_output=False,
                )
                
        elif args.model_type == "SAM": 
            if args.using_BBox:
                if dataset[i]["bounding_box"].shape[0] == 1:
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=dataset[i]["bounding_box"],
                        multimask_output=False,
                    )
                else:
                    input_boxes = torch.from_numpy(dataset[i]["bounding_box"]).to('cuda')
                    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, dataset[i]["image"].shape[:2])
                    masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    )
                    masks = masks.to('cpu')

            else:    
                if dataset[i]["point_coords"].shape[0] == 1:
                    masks, scores, _ = predictor.predict(
                                        point_coords=dataset[i]["point_coords"][0],
                                        point_labels=dataset[i]["point_labels"][0],
                                        multimask_output=False,
                                        )
                else:
                    input_coords = torch.from_numpy(dataset[i]["point_coords"]).to('cuda')
                    input_labels = torch.from_numpy(dataset[i]["point_labels"]).to('cuda')
                    transformed_coords = predictor.transform.apply_coords_torch(input_coords, dataset[i]["image"].shape[:2])
                    masks, scores, _ = predictor.predict_torch(
                                        point_coords=transformed_coords,
                                        point_labels=input_labels,
                                        multimask_output=False,
                                        )
                    masks = masks.to('cpu')
            


        
        if masks.shape[0] > 1:
            for k in range(masks.shape[0]):
                if k == 0:
                    pred_mask = masks[k][0]
                else:
                    pred_mask = (pred_mask + masks[k][0]).clip(0,1)
        else:
            pred_mask = masks[0]
            
            
        (area_intersection[:,i], area_union[:,i], area_sum[:,i]) = intersectionAndUnion(pred_mask, dataset[i]["mask_np"], numClass)
        
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    Dice = 1.0 * np.sum( 2 * area_intersection, axis=1) / np.sum( np.spacing(1) + area_sum, axis=1)
    print(f"mIoU = {IoU} | mDice = {Dice}")


    









def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["kvasir", "CVC_300", "CVC_ClinicDB","CVC_ColonDB", "DDSM", "etis"], default="CVC_300",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["SAM", "SAM2"],
                        default="SAM2",
                        help="Which variant to use.")

    
    parser.add_argument("--n_points_add", default=1, type=int)
    parser.add_argument("--n_points_rmv", default=0, type=int)

    parser.add_argument("--using_BBox", default=True, type=bool)
    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    
    set_seed(args)
    predictor = setup(args)

    if args.dataset == "kvasir":
        dataset  = kvasir(args.n_points_add, args.n_points_rmv, transform=None )
    elif args.dataset == "CVC_300":
        dataset  = CVC_300(args.n_points_add, args.n_points_rmv, transform=None )
    elif args.dataset == "CVC_ClinicDB":
        dataset  = CVC_ClinicDB(args.n_points_add, args.n_points_rmv, transform=None )
    elif args.dataset == "CVC_ColonDB":
        dataset  = CVC_ColonDB(args.n_points_add, args.n_points_rmv, transform=None )
    elif args.dataset == "etis":
        dataset  = etis(args.n_points_add, args.n_points_rmv, transform=None )
        
    Test(args, predictor, dataset)

if __name__ == "__main__":
    main()