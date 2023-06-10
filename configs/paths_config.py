dataset_paths = {
	'ffhq_generate_train': './data/gan_inversion/train',
	'ffhq_generate_test': './data/data/gan_inversion/test',
	'avg_latent_root': '/apdcephfs/share_1290939/kumamzqliu/data/gan_inversion/avg/latent/latent_avg.npy',
	'avg_image_root': '/apdcephfs/share_1290939/kumamzqliu/data/gan_inversion/avg/image/image_avg.png',
	'ffhq_inversion': '/apdcephfs/share_1290939/kumamzqliu/data/face_inversion',


	'car_generate_train': '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion/train',
	'car_generate_test': '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion/test',
	'car_avg_latent_root': '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion/avg/latent/000000.npy',
	'car_avg_image_root': '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion/avg/image/000000.png',
	'car_inversion': '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion',
}

model_paths = {
	'stylegan_ffhq': 'pretrained/stylegan2-ffhq-config-f.pt',
	'stylegan_church': 'pretrained/stylegan2-church-config-f.pt',
	'stylegan_horse': 'pretrained/stylegan2-horse-config-f.pt',
	'ir_se50': 'pretrained/model_ir_se50.pth',
	'circular_face': 'pretrained/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained/pnet.npy',
	'mtcnn_rnet': 'pretrained/rnet.npy',
	'mtcnn_onet': 'pretrained/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained/moco_v2_800ep_pretrain.pt',
	'contrastive_ffhq_image': 'pretrained/ffhq_cont/best_model_image.pt',
	'contrastive_ffhq_latent': 'pretrained/ffhq_cont/best_model_latent.pt',
	'contrastive_car_image': 'pretrained/car_cont/best_model_image.pt',
	'contrastive_car_latent': 'pretrained/car_cont/best_model_latent.pt',

}

edit_paths = {
	'age': 'pretrained/age.pt',
	'pose': '/apdcephfs/share_1290939/kumamzqliu/code/pixel2style2pixel/editings/interfacegan_directions/pose.pt',
	'smile': '/apdcephfs/share_1290939/kumamzqliu/code/pixel2style2pixel/editings/interfacegan_directions/smile.pt',
	'ffhq_pca': '/apdcephfs/share_1290939/kumamzqliu/code/pixel2style2pixel/editings/ganspace_pca/ffhq_pca.pt',
	'car_pca': '/apdcephfs/share_1290939/kumamzqliu/code/pixel2style2pixel/editings/ganspace_pca/cars_pca.pt'
}