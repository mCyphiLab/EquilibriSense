## Run the data collection and visualization
$ python .\gui.py --serial-port COM6

## Evaluate the collected data, missing samples and averange sampling rate
$ python .\evaluate_data.py

## Display the collected data offline
$ python .\visualize_offline.py

## Run the SNS system, on linux since some Windows machine does not support Bluetooth ver 5.2
$ python .\ble.py

## For other analysis and Machine learning, check all jupyter notebooks for the insight