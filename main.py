import matplotlib.pyplot as plt
from minutia_loader import minutiaLoader
from skeleton_maker import skeleton_maker
from reader import load_users
from tqdm import tqdm
from minutiae_filter import minutiae_filter
from load_save import *
import numpy as np

def main():
    data_dir = r"/kaggle/input/10classes/10class" # for kaggle 
    
    # Step 1: Load or process skeleton data
    print("Checking for cached skeleton data...")
    users = load_users_dictionary("processed_data.pkl")
    print(users)
    
    '''
    structure-
    users = {
            "000": {
                "fingers": {
                    "L": [ [impr1, impr2, impr3, impr4, impr5],   # finger 0
                           [impr1, impr2, impr3, impr4, impr5],   # finger 1
                           [impr1, impr2, impr3, impr4, impr5],   # finger 2
                           [impr1, impr2, impr3, impr4, impr5] ], # finger 3
                    "R": [ [...], [...], [...], [...] ]
                }
            },
            ...
        }
    '''
    
    if users is None:
        print("No cached skeleton data found. Processing from scratch...")
        
        # Load original data
        users = load_users(data_dir)
        #print(users)
        #print(users)
        # Process skeletons - Fixed loop structure
        for user_id, user_data  in tqdm(users.items(), desc="Processing users"):
            for hand,fingers in user_data["fingers"].items(): 
                for finger_index, impressions in enumerate(fingers):
                    for impression_index, (image, img_path) in enumerate(impressions):
                        
                            try:
                                fingerprint = minutiaLoader(image)
                                skeleton_image = skeleton_maker(
                                    fingerprint.normalised_img,
                                    fingerprint.segmented_img,
                                    fingerprint.norm_img,
                                    fingerprint.mask,
                                    fingerprint.block,
                                )

                                # Process skeleton
                                skeleton_image.fingerprintPipeline()

                                interim_skeleton = skeleton_image.skeleton
                                interim_minutiae = skeleton_image.minutiae_list
                                interim_mask = fingerprint.mask
                                mf= minutiae_filter(interim_skeleton,interim_minutiae,interim_mask)

                                filtered_fingers, filtered_minutiae = mf.filter_all()


                                users[user_id]["fingers"][hand][finger_index][impression_index] = {
                                        "skeleton": np.array(filtered_fingers),
                                        "minutiae": np.array(filtered_minutiae) ,
                                        "mask": interim_mask,
                                        "orientation_map": skeleton_image.angle_gabor,
                                        "img_path": img_path  
                                    }

                                '''
                                structure-
                                users = {
                                        "000": {
                                            "fingers": {
                                                "L": [ [{
                                                        "skeleton"=[],
                                                        "minutiae"=[],
                                                        "mask"=[],
                                                        "orientation_map"=[]
                                                        },....],   # finger 0
                                                       [impr1, impr2, impr3, impr4, impr5] #finger1,  # finger 2, # finger 3]
                                                "R": [ [...], [...], [...], [...] ]
                                            }
                                        },
                                        ...
                                    }
                                '''



                            except Exception as e:
                                print(f"Error processing {user_id}-{hand}-f{finger_index}-i{impression_index}: {e}")
        #Save processed data
        #print(users)
        save_users_dictionary(users, "processed_data.pkl")
        
    else:
        print("Using cached processed data.")
    

if __name__ == '__main__':
    main()
