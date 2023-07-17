usage: write_obj.py [-h] --model-folder MODEL_FOLDER --motion-file MOTION_FILE
                    --output-folder OUTPUT_FOLDER
                    [--model-type {smpl,smplh,smplx,mano,flame}]
                    [--num-expression-coeffs NUM_EXPRESSION_COEFFS]
                    [--ext EXT] [--sample-expression SAMPLE_EXPRESSION]
                    [--use-face-contour USE_FACE_CONTOUR]

SMPL-X Demo

optional arguments:
  -h, --help            show this help message and exit
  --model-folder MODEL_FOLDER
                        The path to the model folder
  --motion-file MOTION_FILE
                        The path to the motion file to process
  --output-folder OUTPUT_FOLDER
                        The path to the output folder
  --model-type {smpl,smplh,smplx,mano,flame}
                        The type of model to load
  --num-expression-coeffs NUM_EXPRESSION_COEFFS
                        Number of expression coefficients.
  --ext EXT             Which extension to use for loading
  --sample-expression SAMPLE_EXPRESSION
                        Sample a random expression
  --use-face-contour USE_FACE_CONTOUR
                        Compute the contour of the face
