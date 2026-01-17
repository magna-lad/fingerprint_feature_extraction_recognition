import os
import cv2
from collections import defaultdict

#initiate loading the users
#returns path to the users

def load_users(data_dir):
    '''
    expected input:
        - a folder containing several users (000, 001, …)
        - each user has "L" and "R" folders
        - each hand folder contains images like 000_L0_0.bmp … 000_L3_4.bmp
          (4 fingers × 5 impressions = 20 images)

    output:
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
    .....

}
    '''
    users = {}
    for uid in os.listdir(data_dir):
        uid_path = os.path.join(data_dir, uid) # entering the numbered folders
        if not os.path.isdir(uid_path):  # error handling
            continue
            
        users[uid] = {"fingers": {}}
        for hand in ['L', 'R']: # folder structure
            img=[] 
            count=0
            hand_dir = os.path.join(uid_path, hand) 
            if not os.path.isdir(hand_dir):
                continue
            
            if os.path.isdir(hand_dir):
                images=[f for f in os.listdir(hand_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
                images.sort()

                finger_groups = []
            for finger_id in range(4):  #  4 fingers per hand
                start = finger_id * 5
                end = start + 5
                group = []

                for img_name in images[start:end]:
                    img_path = os.path.join(hand_dir, img_name)

                    # Or load with cv2:
                    img = cv2.imread(img_path, 0)  # grayscale
                    group.append((img,img_path))

                finger_groups.append(group)
                users[uid]["fingers"][hand] =  finger_groups
                #print(uid, hand, images)   
    #print(users)
    return users
