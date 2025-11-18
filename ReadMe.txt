Parameter Description
The dataset was generated for the needs of the 3rd International Autonomous Greenhouse Challenge in the experimental facilities of the Greenhouse Horticulture Business Unit in Bleiswijk, 
The Netherlands in 2021. Information regarding the competition can be found here http://www.autonomousgreenhouses.com/.

The dataset contains following folders:
Ground Truth: includes [json] with all GT information
Depth Images: folder with 388  depth images of 16 bit in [png] format 
RGB Images: folder with 388 RGB images of 3x8 bit in [png] format

This dataset contains references to images and measured data on a lettuce crop growing in well controlled greenhouse conditions. There are 4 different lettuce varieties and samples of
the crop are destructively measured in a 7 days interval. 
The lettuce crop varieties are, "Aphylion”, “Salanova”, “Satine” and “Lugano". 
The lettuce plants were grown under different lighting treatments to evaluate the effect of lower and higher light intensities on crop characteristics. However, images of the dataset
are not linked to the individual lighting treatments.
The sampled plants were destructively measured for the following crop traits:
•	FreshWeightShoot [gram]: A head of lettuce, harvested from a hydroponic cultivation system has two parts, the 'root' and the 'shoot' The 'shoot' is the top part, 
	being the edible part of the plant, starting at the attachment point of the first leaves. The trait is measured in gram/plant.
•	Height [cm]: The height of the highest part of the plant, measured from the attachment point of the first leaves. The trait is measured in cm.
•	Diameter [cm]: The principal diameter of the projection of the lettuce plant on a horizontal surface. The trait is measured in cm.
•	LeafArea [cm2]: For the destructive measurements, leaves are torn from the stem and their surface projected on a horizontal surface are marked as LeafArea. 
	Note that curly leaves in reality will have a notable larger area if all curvatures would be straightened. Obviously, the additional surface in these curvatures is discarded with 
	this measurement. The trait is measured in cm2.
•	DryWeightShoot [gram]: After the fresh wait of all above ground parts is weighed and measured, all tissue is dried in a dry-oven. After 3 days of drying the remaining weight is
	measured as dry weight of the shoot. The trait is measured in gram/plant.
Images were taken with one RealSense D415 camera and the dataset includes the collected RGB and Depth data. The depth images were aligned to the color images. Consequently,
the described camera intrinsics of both the depth and color images and resolution (1080x1920) are similar. The RGB images [3x8bit] are stored in the 'RGB Images' folder
and have a [png] extension. The depth images are stored in the 'Depth Images' folder. These images are [16bit] and have a [png] extension. The depth images can be converted to
3D point clouds using the intrinsics below or the intrinsics in GT json. 

The 'GroundTruth_All_388_Images.json' file contains all ground truth information. The corresponding RGB and depth image can be found using the names of each measurement ("RGBImage" and
"DebthInformation"


DeviceID: 002422060160
DepthScale: 0.00100000004749745
depth_int:
•	height: 1080
•	width: 1920
•	ppx: 973.902038574219
•	ppy: 537.702270507812
•	fx: 1371.58264160156
•	fy: 1369.42761230469
•	coefs: [ 0.0, 0.0, 0.0, 0.0, 0.0 ]
•	model: "distortion.inverse_brown_conrady"
color_int:
•	height: 1080
•	width: 1920
•	ppx: 973.902038574219
•	ppy: 537.702270507812
•	fx: 1371.58264160156
•	fy: 1369.42761230469
•	coefs: [ 0.0, 0.0, 0.0, 0.0, 0.0 ]
•	model: "distortion.inverse_brown_conrady"
ir_int:
•	height" : 720
•	width" : 1280
•	ppx: 637.933654785156
•	ppy: 347.102325439453
•	fx: 889.749572753906
•	fy: 889.749572753906
•	coefs: [ 0.0, 0.0, 0.0, 0.0, 0.0 ]
•	"model" : "distortion.brown_conrady"
depth_to_color_ext:
•	rotation: [ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ]
•	translation: [ 0.0, 0.0, 0.0 ]
ir_to_color_ext:
•	rotation: [0.999989449977875,	
  -0.00311925541609526,
  0.00337212649174035,
  0.00313232676126063,
  0.999987602233887,
  -0.00387795572169125,
  -0.00335998833179474,
  0.00388847757130861,
  0.99998676776886]
•	translation: [ 0.0151169626042247, -2.81767242995556e-05, -0.000219923706026748 ]
