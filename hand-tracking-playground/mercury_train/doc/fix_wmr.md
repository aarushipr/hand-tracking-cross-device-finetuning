- Windows Mixed Reality shows three distinct failure modes I hadn't identified before:  
  - Too much contrast - Either the hand is so bright that it washes out the sensor cells and mostly is 255, or the background is so bright that the hand is mostly 0  
  - Not enough contrast - The hand happens to have almost the same luminance as the background and it's hard to track  
  - Too dark - WMR's "tracking" autoexposure strategy results in a hand that's too dark\* (imagine the histogram of luminance values coming out of the hand is like 0-16 or whatever.) This confuses the heck out of our NN - it's never seen images with a low number of discrete luminance values. The vast majority of the time, up until now, it'd see like*at least* 100 discrete values, if not all 255.  
  - Note: the images are also grainy. We already have data augmentations for this and *I don't think camera grain is the problem here.*  
- I propose *two* data augmentations to combat these. Each one would run say 5% of the time:  
  - 1 - characterizing "too much contrast":  
    - Pick random upper and lower bounds in 0-255 uint8-space to clamp image to  
    - Clamp the image to these  
    - Convert 0-255 to 0-1. The lowest value will be higher than 0 and the highest will be lower than 1  
    - Send onward. It will eventually be normalized to 0.5 mean and 0.25 stddev by a later part of the data pipeline.  
    - (You could also do this using a 1D homothety and clamp to 0-255, but that would do basically the same thing)  
  - 2 - characterizing "too dark" *and* "too little contrast" (these are actually the same thing when you consider that our neural nets are convolutional and only really see differences between pixels)  
    - Pick a target mean between like 2 and 128. (Probably use a stepwise distribution so you can have only a small number actually be 2, the peak around 64, etc.)  
    - In 0-255 space, multiply the input image by (target_mean)/(current_mean). This will greatly reduce the number of discrete luminances and make features harder to see  
    - Send onward. It will eventually be normalized to 0.5 mean and 0.25 stddev by a later part of the data pipeline.  
-  
  - \* - It's this code snippet:  

  ```
   if (aeg->strategy == U_AEG_STRATEGY_TRACKING) {
    // We are not that much interested in using the full dynamic range for tracking
    // so we prefer a darkish image because that reduces exposure and gain.
    target_mean = LEVELS / 4;
   }
  ```

  Was hard to find because I was grepping for "64" and not X / 4.  
- Cool idea that's probably unnecessary:  
  - Send hand regions of interest back to the AEG module, target mean of 96, and lerp between what the AEG wants to do for the full image and what it wants to do for just the .  




mateo:
     Regarding generating contrast: If you want something that works as well as a "contrast curve" from a image editing software, use something like [this](https://forum.unity.com/threads/shader-function-to-adjust-texture-contrast.457635/). I feel clamping sounds like a bad idea for just contrast
     
     Regardging the "cool idea" at the end. Is there even any value when doing manual exposure/gain that is good enough exposure-gain balance for both slam and hand tracking? if there's maybe you can turn that into another heuristic/strategy in the aeg module

jakob:

Some feature requests that aren't urgent but I don't want to lose.

    Env var to select onnx model(s)
    Recorder-tron-3000++
        Keep a configurable amount of frames memory
        Dump out when pressing a button or triggered by code
        Configurable to wait X time before dumping
    Train compatible model with full set of models (non-distributable)