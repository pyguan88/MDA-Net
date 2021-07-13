import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio
# from scipy.misc import imresize
import copy
from Spa_downs import *


names_CAVE = [
    'balloons_ms','thread_spools_ms', 'fake_and_real_food_ms', 'face_ms','feathers_ms', 'cd_ms', 'real_and_fake_peppers_ms',
    'stuffed_toys_ms', 'sponges_ms', 'oil_painting_ms', 'fake_and_real_strawberries_ms', 'fake_and_real_beers_ms',
    'fake_and_real_lemon_slices_ms', 'fake_and_real_sushi_ms', 'egyptian_statue_ms', 'glass_tiles_ms', 'jelly_beans_ms',
    'fake_and_real_peppers_ms', 'clay_ms', 'pompoms_ms', 'watercolors_ms', 'fake_and_real_tomatoes_ms'
    ]

names_CAVE_valid = ['real_and_fake_apples_ms', 'superballs_ms', 'chart_and_stuffed_toy_ms', 'hairs_ms',  'fake_and_real_lemons_ms',
    'flowers_ms', 'paints_ms', 'photo_and_face_ms', 'cloth_ms', 'beads_ms'
]

names_Harvard = [
    'imge6', 'imgc4', 'imgf8', 'imgb7', 'imgd4', 'imgb1', 'imge1', 'imga6', 'imgh2', 'imgb3', 'imgf3',
    'imgf4', 'imge0', 'imgd3', 'img2', 'imgf2', 'imge5', 'imgc8', 'imge2', 'imgc7', 'imgb9', 'imgh3',
    'imgc5', 'imga7', 'imgb4', 'imgh0', 'imgd7', 'imge7', 'imgb6', 'imga5', 'imgf7', 'imgc2', 'imgf5',
    'imgb2', 'imge3', 'imgc1', 'imga1', 'imgc9', 'imgb5', 'img1', 'imgb0', 'imgd8', 'imgb8'
]

names_ICLV = [
    'BGU_HS_00001','BGU_HS_00030','BGU_HS_00060','BGU_HS_00090','BGU_HS_00120','BGU_HS_00150','BGU_HS_00180',
    'BGU_HS_00020', 'BGU_HS_00040', 'BGU_HS_00050', 'BGU_HS_00070', 'BGU_HS_00080', 'BGU_HS_00100', 'BGU_HS_00110',
    'BGU_HS_00130', 'BGU_HS_00140', 'BGU_HS_00160', 'BGU_HS_00170', 'BGU_HS_00190', 'BGU_HS_00200','BGU_HS_00010',
]

class LoadDataset(data.Dataset):
    def __init__(self, Path, datasets='CAVE', patch_size=128, stride=64, Data_Aug=False, up_mode='bicubic', Train_image_num=22):
        super(LoadDataset, self).__init__()

        if datasets == 'CAVE':
            self.names = names_CAVE
        elif datasets == 'Harvard':
            self.names = names_Harvard
        elif datasets == 'ICLV':
            self.names = names_ICLV
        elif datasets == 'CAVE_val':
            self.names = names_CAVE_valid
        else:
            assert 'wrong dataset name'

        self.path = Path                                #The path of HR HSI
        self.Image_size = 512                           #The size of original HR HSI
        self.P_S = patch_size                           #We devide the HR HSI into patches at first and this indicate the size of patch
        self.stride = stride                            #The stride of each patch.
        self.DA = Data_Aug                              #Use the data augmentation or not.
        self.P_N = int(self.Image_size/self.stride)     #The number of patches.
        self.up_mode = up_mode                          #The upsample mode.
        self.img_num = Train_image_num                  #The number of images in training set.
        # self.gff_path = Path.replace('CAVE', 'CAVE_GFF_X8')


    def __getitem__(self, Index):


        P_S = self.P_S
        S = self.stride
        P_N = self.P_N

        if self.DA:
            Aug = 2
        else:
            Aug = 1


        Image_size = self.Image_size
        Patches = P_N**2
        Image_I = int(Index/Aug/Patches)
        Patch_I = int(Index/Aug%Patches)

        Data = sio.loadmat(self.path+self.names[Image_I]+'.mat')
        # GFF_Data = sio.loadmat(self.gff_path + self.names[Image_I]+'.mat')


        HSI = Data['hsi']
        # GFF_HSI = Data['gff_hsi']

        HSI = (HSI/(np.max(HSI)-np.min(HSI))).transpose((2,0,1))
        # GFF_HSI = (GFF_HSI/(np.max(GFF_HSI)-np.min(GFF_HSI))).transpose((2,0,1))



        #for i in range(HSI.shape[0]):
            # nearest,lanczos,bilinear,bicubic,cubic
        #    HSI_Up[i,:,:] = imresize(HSI[i,:,:], (MSI.shape[1], MSI.shape[2]), self.up_mode, mode='F' )

        X = int(Patch_I/P_N) #X,Y is patch index in image
        Y = int(Patch_I%P_N)

        s = int(S/8)       ### set the scal factor as 8
        p_s = int(P_S/8)

        if X*S+P_S > Image_size and Y*S+P_S <= Image_size:
            GT = HSI[:, -P_S:, Y * S: Y * S + P_S]
            # INIT = GFF_HSI[:, -P_S:, Y * S: Y * S + P_S]
        elif X*S+P_S <= Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, X * S:X * S + P_S, -P_S:]
            # INIT = GFF_HSI[:, X * S:X * S + P_S, -P_S:]
        elif X*S+P_S > Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, -P_S: , -P_S: ]
            # INIT = GFF_HSI[:, -P_S: , -P_S: ]
        else:
            GT = HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]
            # INIT = GFF_HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]


        # Data augmantation
        if self.DA :
            if Index%2 == 1:
                a = np.random.randint(0,6,1)
                if a[0] == 0:
                    GT = copy.deepcopy(np.flip(GT, 1))  # flip the array upside down
                    # INIT = copy.deepcopy(np.flip(INIT, 1))  # flip the array upside down
                elif a[0] == 1:
                    GT = copy.deepcopy(np.flip(GT, 2))  # flip the array left to right
                    # INIT = copy.deepcopy(np.flip(INIT, 2))  # flip the array left to right
                elif a[0] == 2:
                    GT = copy.deepcopy(np.rot90(GT, 1, [1, 2]))  # Rotate 90 degrees clockwise
                    # INIT = copy.deepcopy(np.rot90(INIT, 1, [1, 2]))  # Rotate 90 degrees clockwise
                elif a[0] == 3:
                    GT = copy.deepcopy(np.rot90(GT, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
                    # INIT = copy.deepcopy(np.rot90(INIT, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
                elif a[0] == 4:
                    GT = copy.deepcopy(np.roll(GT, int(GT.shape[1] / 2), 1))  # Roll the array up
                    # INIT = copy.deepcopy(np.roll(INIT, int(INIT.shape[1] / 2), 1))  # Roll the array up
                elif a[0] == 5:
                    GT = np.roll(GT, int(GT.shape[1] / 2), 1)  # Roll the array up & left
                    GT = copy.deepcopy(np.roll(GT, int(GT.shape[2] / 2), 2))
                    # INIT = np.roll(INIT, int(INIT.shape[1] / 2), 1)  # Roll the array up & left
                    # INIT = copy.deepcopy(np.roll(INIT, int(INIT.shape[2] / 2), 2))

        GT = torch.from_numpy(GT)
        # INIT = torch.from_numpy(INIT)

        return GT

    def __len__(self):

        if self.DA:
            Aug = 2
        else:
            Aug = 1

        return int(self.P_N**2*self.img_num*Aug)


class LoadDataset_Mem(data.Dataset):
    def __init__(self, allData, patch_size=128, stride=64, Data_Aug=1):
        super(LoadDataset_Mem, self).__init__()


        self.data = allData                             #The list of all data
        self.Image_size = allData[-1].shape[2]           #The size of original HR HSI
        self.P_S = patch_size                           #We devide the HR HSI into patches at first and this indicate the size of patch
        self.stride = stride                            #The stride of each patch.
        self.DA = Data_Aug                              #Use the data augmentation or not.
        self.P_N = int(self.Image_size/self.stride)     #The number of patches.


    def __getitem__(self, Index):

        P_S = self.P_S
        S = self.stride
        P_N = self.P_N

        # if self.DA:
        #     Aug = 2
        # else:
        #     Aug = 1
        Aug = self.DA

        Image_size = self.Image_size
        Patches = P_N**2
        Image_I = int(Index/Aug/Patches)
        Patch_I = int(Index/Aug%Patches)

        HSI = self.data[Image_I]
        # GFF_Data = sio.loadmat(self.gff_path + self.names[Image_I]+'.mat')


        # HSI = Data['hsi']
        # GFF_HSI = Data['gff_hsi']

        # HSI = (HSI/(np.max(HSI)-np.min(HSI))).transpose((2,0,1))
        # GFF_HSI = (GFF_HSI/(np.max(GFF_HSI)-np.min(GFF_HSI))).transpose((2,0,1))



        #for i in range(HSI.shape[0]):
            # nearest,lanczos,bilinear,bicubic,cubic
        #    HSI_Up[i,:,:] = imresize(HSI[i,:,:], (MSI.shape[1], MSI.shape[2]), self.up_mode, mode='F' )

        X = int(Patch_I/P_N) #X,Y is patch index in image
        Y = int(Patch_I%P_N)

        s = int(S/8)       ### set the scal factor as 8
        p_s = int(P_S/8)

        if X*S+P_S > Image_size and Y*S+P_S <= Image_size:
            GT = HSI[:, -P_S:, Y * S: Y * S + P_S]
            # INIT = GFF_HSI[:, -P_S:, Y * S: Y * S + P_S]
        elif X*S+P_S <= Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, X * S:X * S + P_S, -P_S:]
            # INIT = GFF_HSI[:, X * S:X * S + P_S, -P_S:]
        elif X*S+P_S > Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, -P_S: , -P_S: ]
            # INIT = GFF_HSI[:, -P_S: , -P_S: ]
        else:
            GT = HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]
            # INIT = GFF_HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]


        # Data augmantation
        if Index%Aug != 0:
            a = np.random.randint(0,6,1)
            if a[0] == 0:
                GT = copy.deepcopy(np.flip(GT, 1))  # flip the array upside down
                # INIT = copy.deepcopy(np.flip(INIT, 1))  # flip the array upside down
            elif a[0] == 1:
                GT = copy.deepcopy(np.flip(GT, 2))  # flip the array left to right
                # INIT = copy.deepcopy(np.flip(INIT, 2))  # flip the array left to right
            elif a[0] == 2:
                GT = copy.deepcopy(np.rot90(GT, 1, [1, 2]))  # Rotate 90 degrees clockwise
                # INIT = copy.deepcopy(np.rot90(INIT, 1, [1, 2]))  # Rotate 90 degrees clockwise
            elif a[0] == 3:
                GT = copy.deepcopy(np.rot90(GT, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
                # INIT = copy.deepcopy(np.rot90(INIT, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
            elif a[0] == 4:
                GT = copy.deepcopy(np.roll(GT, int(GT.shape[1] / 2), 1))  # Roll the array up
                # INIT = copy.deepcopy(np.roll(INIT, int(INIT.shape[1] / 2), 1))  # Roll the array up
            elif a[0] == 5:
                GT = np.roll(GT, int(GT.shape[1] / 2), 1)  # Roll the array up & left
                GT = copy.deepcopy(np.roll(GT, int(GT.shape[2] / 2), 2))
                # INIT = np.roll(INIT, int(INIT.shape[1] / 2), 1)  # Roll the array up & left
                # INIT = copy.deepcopy(np.roll(INIT, int(INIT.shape[2] / 2), 2))

        GT = torch.from_numpy(GT)
        # INIT = torch.from_numpy(INIT)

        return GT

    def __len__(self):

        # if self.DA:
        #     Aug = 2
        # else:
        #     Aug = 1
        Aug = self.DA
        return int(self.P_N**2*len(self.data)*Aug)


class Val_LoadDataset(data.Dataset):
    def __init__(self, Path, datasets='CAVE_val',  Train_image_num=5):
        super(Val_LoadDataset, self).__init__()

        if datasets == 'CAVE_val':
            self.names = names_CAVE_valid
        else:
            assert 'wrong dataset name'

        self.path = Path                                #The path of HR HSI
        self.img_num = Train_image_num

    def __getitem__(self, Index):

        Data = sio.loadmat(self.path+self.names[Index]+'.mat')
        HSI = Data['hsi']
        HSI = (HSI/(np.max(HSI)-np.min(HSI))).transpose((2,0,1))

        return torch.from_numpy(HSI)

    def __len__(self):
        return self.img_num


class LoadDataset_Mem_Val(data.Dataset):
    def __init__(self, allValData):
        super(LoadDataset_Mem_Val, self).__init__()
        self.data = allValData                             #The list of all data

    def __getitem__(self, Index):

        HSI = self.data[Index]
        return torch.from_numpy(HSI)

    def __len__(self):

        return len(self.data)

class LoadRealDataset_Mem_Val(data.Dataset):
    def __init__(self, allValMsi, allValPan):
        super(LoadRealDataset_Mem_Val, self).__init__()
        self.MsiData = allValMsi
        self.PanData = allValPan

    def __getitem__(self, Index):

        MSI = self.MsiData[Index]
        PAN = self.PanData[Index]
        return torch.from_numpy(MSI), torch.from_numpy(PAN)

    def __len__(self):
        assert len(self.MsiData) == len(self.PanData), 'Unequal number of msi and pan data.'
        return len(self.MsiData)

class LoadRealDataset_Mem(data.Dataset):
    def __init__(self, allMsi, allPan, patch_size=128, stride=64, Data_Aug=1):
        super(LoadRealDataset_Mem, self).__init__()


        self.MsiData = allMsi                             #The list of all data
        self.PanData = allPan
        self.Image_size = allPan[-1].shape[2]           #The size of original HR Pan
        self.P_S = patch_size                           #We devide the HR HSI into patches at first and this indicate the size of patch
        self.stride = stride                            #The stride of each patch.
        self.DA = Data_Aug                              #Use the data augmentation or not.
        self.P_N = int(self.Image_size/self.stride)     #The number of patches.
        assert patch_size%4==0 and stride%4==0, 'The patch size or the stride is not devided by 4.'

    def __getitem__(self, Index):

        P_S = self.P_S
        S = self.stride
        P_N = self.P_N

        Aug = self.DA

        Image_size = self.Image_size
        Patches = P_N**2
        Image_I = int(Index/Aug/Patches)
        Patch_I = int(Index/Aug%Patches)

        PAN = self.PanData[Image_I]
        MSI = self.MsiData[Image_I]

        X = int(Patch_I/P_N) #X,Y is patch index in image
        Y = int(Patch_I%P_N)

        s = int(S/4)       ### set the scal factor as 8
        p_s = int(P_S/4)

        if X*S+P_S > Image_size and Y*S+P_S <= Image_size:
            real_pan = PAN[:, -P_S:, Y * S: Y * S + P_S]
            real_msi = MSI[:, -p_s:, Y * s: Y * s + p_s]
        elif X*S+P_S <= Image_size and Y*S+P_S > Image_size:
            real_pan = PAN[:, X * S:X * S + P_S, -P_S:]
            real_msi = MSI[:, X * s:X * s + p_s, -p_s:]
        elif X*S+P_S > Image_size and Y*S+P_S > Image_size:
            real_pan = PAN[:, -P_S: , -P_S: ]
            real_msi = MSI[:, -p_s: , -p_s: ]
        else:
            real_pan = PAN[:, X * S:X * S + P_S, Y * S:Y * S + P_S]
            real_msi = MSI[:, X * s:X * s + p_s, Y * s:Y * s + p_s]

        # Data augmantation
        if Index%Aug != 0:
            a = np.random.randint(0,6,1)
            if a[0] == 0:
                real_pan = copy.deepcopy(np.flip(real_pan, 1))  # flip the array upside down
                real_msi = copy.deepcopy(np.flip(real_msi, 1))  # flip the array upside down
                # INIT = copy.deepcopy(np.flip(INIT, 1))  # flip the array upside down
            elif a[0] == 1:
                real_pan = copy.deepcopy(np.flip(real_pan, 2))  # flip the array left to right
                real_msi = copy.deepcopy(np.flip(real_msi, 2))  # flip the array left to right
            elif a[0] == 2:
                real_pan = copy.deepcopy(np.rot90(real_pan, 1, [1, 2]))  # Rotate 90 degrees clockwise
                real_msi = copy.deepcopy(np.rot90(real_msi, 1, [1, 2]))
            elif a[0] == 3:
                real_pan = copy.deepcopy(np.rot90(real_pan, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
                real_msi = copy.deepcopy(np.rot90(real_msi, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
            elif a[0] == 4:
                real_pan = copy.deepcopy(np.roll(real_pan, int(real_pan.shape[1] / 2), 1))  # Roll the array up
                real_msi = copy.deepcopy(np.roll(real_msi, int(real_msi.shape[1] / 2), 1))  # Roll the array up
            elif a[0] == 5:
                real_pan = np.roll(real_pan, int(real_pan.shape[1] / 2), 1)  # Roll the array up & left
                real_pan = copy.deepcopy(np.roll(real_pan, int(real_pan.shape[2] / 2), 2))
                real_msi = np.roll(real_msi, int(real_msi.shape[1] / 2), 1)  # Roll the array up & left
                real_msi = copy.deepcopy(np.roll(real_msi, int(real_msi.shape[2] / 2), 2))
        real_pan = torch.from_numpy(real_pan)
        real_msi = torch.from_numpy(real_msi)

        return real_msi, real_pan

    def __len__(self):

        # if self.DA:
        #     Aug = 2
        # else:
        #     Aug = 1
        Aug = self.DA
        return int(self.P_N**2*len(self.PanData)*Aug)
