# __Getting Started Application Code__


For the STM32N6, following package provides STM32 real-time applications to execute [STEdgeAI](https://www.st.com/en/development-tools/stedgeai-core.html) generated models targeting different use cases.
Applications are made to be used in a standalone way. But can be as well used in the [ModelZoo](https://github.com/STMicroelectronics/stm32ai-modelzoo-services).

It is also important to note that that the application code for the STM32N6 shall be downloaded from [https://www.st.com/en/development-tools/stm32n6-ai.html](https://www.st.com/en/development-tools/stm32n6-ai.html) and unzipped in the application code.
The STM32N6 provided applications are:
- [Image classification](image_classification/STM32N6/README.md)
- [Object detection](object_detection/STM32N6/README.md)
- [Pose estimation](pose_estimation/STM32N6/README.md)
- [Instance segmentation](instance_segmentation/STM32N6/README.md)
- [Semantic segmentation](semantic_segmentation/STM32N6/README.md)
- [Audio event detection](audio/STM32N6/README.md)
- [Speech enhancement](audio/STM32N6/README.md)

You can use the [ST ModelZoo repo](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) to deploy the use cases on STM32N6. This package is needed by the ModelZoo for STM32N6 Target. The ModelZoo allows you to train, evaluate and deploy automatically any supported model. If you wish to use it as part of the ModelZoo, please refer to the [ModelZoo README](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/README.md) on how to use it.


For the other MCU and MPU platform, provided applications are documented below:
- [Image classification on STM32H7](image_classification/STM32H7/README.md)
- [Image classification on MPU](image_classification/STM32MP-LINUX/README.md)
- [Object detection on STM32H7](object_detection/STM32H7/README.md)
- [Object detection on MPU](object_detection/STM32MP-LINUX/README.md)
- [Pose estimation on MPU](pose_estimation/STM32MP-LINUX/README.md)
- [Semantic segmentation on MPU](semantic_segmentation/STM32MP-LINUX/README.md)
- [Human activity recognition on STM32U5](sensing/STM32U5/README.md)
- [Audio event detection on STM32U5](sensing/STM32U5/README.md)
- [Hand posture on STM32F4](hand_posture/STM32F4/README.md)


