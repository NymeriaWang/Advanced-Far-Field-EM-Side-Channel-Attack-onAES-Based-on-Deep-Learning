# Advanced-Far-Field-EM-Side-Channel-Attack-onAES-Based-on-Deep-Learning
The reporitory contains scripts and data used in the paper &lt;Advanced Far Field EM Side-Channel Attack on AES Based on Deep Learning>. This paper presents a deep learning-based side-channel attack on AES-128 using far field electromagnetic (EM) emissions as a side channel. Unlike power or near filed EM analysis, far field EM attacks do not require a close physical proximity to the device under attack. Our neural networks are trained on traces captured from five different Bluetooth devices and tested on five other Bluetooth devices. The training set is composed as a combination of  ``clean'' traces, captured through a coaxial cable, and traces captured on a distance to device. Our best model can recover the key from less than 300 traces captured in an office environment at 15 m distance to device without repeating each encryption more than once. 

Please find our data in https://drive.google.com/drive/folders/1gikCpMKPnBBFnhFj4Yfga9Tufeyx4GFZ?usp=sharing 

The folder called training_data contains 200k EM traces captured from 5 Bluetooth-supported nRF52 devices. Each trace is captured by coaxial cable from the profiling device and each trace is the average of 100 traces of the same encryption. 

The folder called testing_data contains 25k EM traces captured from another 5 Bluetooth-supported nRF52 devices. Each trace is captured at 15 meters distance to the victim device. For each device, we further have 1, 10 and 100_avg section which are for the case that a testing trace is the average of 1, 10 and 100 traces of the same encryption. 

For further details, please check our work 'Wang, Ruize, et al. "Advanced Far Field EM Side-Channel Attack on AES." Proceedings of the 7th ACM on Cyber-Physical System Security Workshop. 2021.'
