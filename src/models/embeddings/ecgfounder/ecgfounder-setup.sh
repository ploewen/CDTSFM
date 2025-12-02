git clone https://github.com/PKUDigitalHealth/ECGFounder.git

mv ECGFounder/net1d.py src/models/embeddings/ecgfounder/net1d.py

rm -rf ECGFounder

wget https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/1_lead_ECGFounder.pth -O weights/1_lead_ECGFounder.pth