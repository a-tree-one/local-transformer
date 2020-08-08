
# A New Local Transformation Module for Few-shot Segmentation
[paper](https://arxiv.org/abs/1910.05886v1)
## Abstract
Few-shot segmentation segments object regions of new classes with a few of manual annotations. Its key step is to establish the transformation module between support images (annotated images) and query images (unlabeled images), so that the segmentation cues of support images can guide the segmentation of query images. The existing methods form transformation model based on global cues, which however ignores the local cues that are verified in this paper to be very important for the transformation. This paper proposes a new transformation module based on local cues, where the relationship of the local features is used for transformation. To enhance the generalization performance of the network, the relationship matrix is calculated in a high-dimensional metric embedding space based on cosine distance. In addition, to handle the challenging mapping problem from the low-level local relationships to high-level semantic cues, we propose to apply generalized inverse matrix of the annotation matrix of support images to transform the relationship matrix linearly, which is non-parametric and class-agnostic. The result by the matrix transformation can be regarded as an attention map with high-level semantic cues, based on which a transformation module can be built simply.The proposed transformation module is a general module that can be used to replace the transformation module in the existing few-shot segmentation frameworks. We verify the effectiveness of the proposed method on Pascal VOC 2012 dataset. The value of mIoU achieves at 57.0% in 1-shot and 60.6% in 5-shot, which outperforms the state-of-the-art method by 1.6% and 3.5%, respectively.
## Implemtation
1) the code is based on cv2, pytorch, pydensecrf, numpy, json
2) In order to create the images pairs of training, the file of json is generated
3) In order to create the images pairs of validation, the dir of  "sbd" is created, and it includes three parts: 
  1.the dir of images: "img" 2. the dir of the ground truth: "cls" 3. the "val.txt"
## The Proposed Network
Different to the existing methods that global features are used to realize the guided segmentation, the local features are considered in this paper, which can transfer the local and structual cues contained in supported mask.
> Network
![image](https://github.com/a-tree-one/local-transformer/blob/master/network.png)


