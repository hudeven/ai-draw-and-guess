# Model Serving with torchserve

## Prepare modelstore
TODO: provide link to the original dalle mini checkpoint files and CLI to convert them to .mar files 

## Start torchserve
`cd ai-draw-and-guess/serve/`
`torchserve --start --ncs --model-store modelstore --models dalle_mega.mar,dalle_mini.mar --ts-config config.properties`
