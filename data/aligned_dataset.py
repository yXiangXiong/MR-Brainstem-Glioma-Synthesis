import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input T1, T2, ASL (label maps)
        dir_T1 = 'T1' if self.opt.label_nc == 0 else '_inst'
        dir_T2 = 'T2' if self.opt.label_nc == 0 else '_inst'
        dir_ASL = 'ASL' if self.opt.label_nc == 0 else '_inst'
        self.dir_T1 = os.path.join(opt.dataroot, opt.phase + dir_T1)
        self.dir_T2 = os.path.join(opt.dataroot, opt.phase + dir_T2)
        self.dir_ASL = os.path.join(opt.dataroot, opt.phase + dir_ASL)
        self.T1_paths = sorted(make_dataset(self.dir_T1))
        self.T2_paths = sorted(make_dataset(self.dir_T2))
        self.ASL_paths = sorted(make_dataset(self.dir_ASL))

        ### T1ce, tumor mask (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_Cplus = 'Cplus' if self.opt.label_nc == 0 else '_label'
            self.dir_Cplus = os.path.join(opt.dataroot, opt.phase + dir_Cplus)
            self.Cplus_paths = sorted(make_dataset(self.dir_Cplus))
            dir_Mask = 'Mask'
            self.dir_Mask = os.path.join(opt.dataroot, opt.phase + dir_Mask)
            self.Mask_paths = sorted(make_dataset(self.dir_Mask))
        else:
            dir_Cplus = 'Cplus' if self.opt.label_nc == 0 else '_label'
            self.dir_Cplus = os.path.join(opt.dataroot, opt.phase + dir_Cplus)
            self.Cplus_paths = sorted(make_dataset(self.dir_Cplus))
            dir_Mask = 'Mask'
            self.dir_Mask = os.path.join(opt.dataroot, opt.phase + dir_Mask)
            self.Mask_paths = sorted(make_dataset(self.dir_Mask))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.T1_paths)


    def __getitem__(self, index):
        T1_path = self.T1_paths[index]
        T2_path = self.T2_paths[index]
        ASL_path = self.ASL_paths[index]
        T1 = Image.open(T1_path).convert('RGB')
        T2 = Image.open(T2_path).convert('RGB')
        ASL = Image.open(ASL_path).convert('RGB')
        params = get_params(self.opt, T1.size)
        if self.opt.label_nc == 0:
            transform_ = get_transform(self.opt, params)
            T1_tensor = transform_(T1.convert('RGB'))
            T2_tensor = transform_(T2.convert('RGB'))
            ASL_tensor = transform_(ASL.convert('RGB'))
        else:
            transform_ = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            T1_tensor = transform_(T1) * 255.0
            T2_tensor = transform_(T2) * 255.0
            ASL_tensor = transform_(ASL) * 255.0
        label_tensor = torch.cat((T1_tensor, T2_tensor, ASL_tensor))

        Cplus_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            Cplus_path = self.Cplus_paths[index]
            Cplus = Image.open(Cplus_path).convert('RGB')
            transform_Cplus = get_transform(self.opt, params)
            Cplus_tensor = transform_Cplus(Cplus)

            Mask_path = self.Mask_paths[index]
            Mask = Image.open(Mask_path).convert('RGB')
            transform_Mask = get_transform(self.opt, params)
            Mask_tensor = transform_Mask(Mask)

            image_tensor = torch.cat((Cplus_tensor, Mask_tensor))
        else:
            Cplus_path = self.Cplus_paths[index]
            Cplus = Image.open(Cplus_path).convert('RGB')
            transform_Cplus = get_transform(self.opt, params)
            Cplus_tensor = transform_Cplus(Cplus)

            Mask_path = self.Mask_paths[index]
            Mask = Image.open(Mask_path).convert('RGB')
            transform_Mask = get_transform(self.opt, params)
            Mask_tensor = transform_Mask(Mask)

            image_tensor = torch.cat((Cplus_tensor, Mask_tensor))

        input_dict = {'label': label_tensor, 'inst': inst_tensor, 'image': image_tensor,
                      'feat': feat_tensor, 'path': T1_path}

        return input_dict

    def __len__(self):
        return len(self.T1_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'