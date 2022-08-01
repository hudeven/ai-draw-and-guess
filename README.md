# ai-draw-and-guess

## Install dependencies
`pip install -r requirements.txt`

## Start server
### for dev purpose
`python main.py`
### for production
`gunicorn -w 1 --threads 100 main:app -b 0.0.0.0:9000`

## Start clients
Visit http://127.0.0.1:9000 in 2 or more tabs in your browser 
