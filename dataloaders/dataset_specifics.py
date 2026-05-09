"""
Dataset Specifics
"""

def get_label_names(dataset):
    label_names = {}
    if dataset == 'CARDIAC_bssFP':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'
    elif dataset == 'CARDIAC_LGE':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'
    elif dataset == 'ABDOMEN_MR':
        label_names[0] = 'BG'
        label_names[1] = 'LIVER'
        label_names[2] = 'RIGHT_KIDNEY'
        label_names[3] = 'LEFT_KIDNEY'
        label_names[4] = 'SPLEEN'        
    elif dataset == 'ABDOMEN_CT':
        label_names[0] = 'BG'
        label_names[1] = 'SPLEEN'
        label_names[2] = 'RIGHT_KIDNEY'
        label_names[3] = 'LEFT_KIDNEY'
        label_names[4] = 'GALLBLADDER'
        label_names[5] = 'ESOPHAGUS'
        label_names[6] = 'LIVER'
        label_names[7] = 'STOMACH'
        label_names[8] = 'AORTA'
        label_names[9] = 'INFERIOR_VENA_CAVA'             # Inferior vena cava
        label_names[10] = 'PORTAL_VEIN_AND_SPLENIC_VEIN'  # portal vein and splenic vein
        label_names[11] = 'PANCREAS'
        label_names[12] = 'RIGHT_ADRENAL_GLAND'  # right adrenal gland
        label_names[13] = 'LEFT_ADRENAL_GLAND'   # left adrenal gland
    elif dataset == 'MI-PRO':
        label_names[0] = 'BG'
        label_names[1] = 'Bladder'
        label_names[2] = 'Bone'
        label_names[3] = 'Obturator_Internus'
        label_names[4] = 'Transition_Zone'
        label_names[5] = 'Central_Gland'
        label_names[6] = 'Rectum'
        label_names[7] = 'Seminal_Vesicle'
        label_names[8] = 'Neurovascular_Bundle'
    return label_names


