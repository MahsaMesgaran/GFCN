# GFCN : Graph Fairing Convolutional Networks for Anomaly Detection

Graph convolution is a fundamental building block for many deep neural networks on graph-structured data.
In this paper, we introduce a simple, yet very effective graph convolutional network with skip connections for
semi-supervised anomaly detection. The proposed multi-layer network architecture is theoretically motivated by
the concept of implicit fairing in geometry processing, and comprises a graph convolution module for aggregating
information from immediate node neighbors and a skip connection module for combining layer-wise neighborhood
representations. In addition to capturing information from distant graph nodes through skip connections between the
network’s layers, our approach exploits both the graph structure and node features for learning discriminative node
representations. The effectiveness of our model is demonstrated through extensive experiments on five benchmark
datasets, achieving better or comparable anomaly detection results against strong baseline methods.


# Requirements
scipy==1.7.3
python==3.6.1
matplotlib==3.3.4
numpy==1.19.2
scikit-learn==0.24.1
scipy==1.5.2
torch==1.8.1

