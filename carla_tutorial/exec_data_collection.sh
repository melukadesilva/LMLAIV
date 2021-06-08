cd /opt/carla-simulator/PythonAPI/util
python3 config.py  --map Town07
nohup python3 environment.py --clouds 100 --rain 80 --wetness 100 --puddles 60 --wind 80 --fog 5
cd ../examples/
nohup python3 dynamic_weather.py --speed 1.0 &
nohup python3 spawn_npc.py -n 50 -w 50 --safe &
