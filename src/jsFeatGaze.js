(function(window) {
    //"use strict";

    window.webgazer = window.webgazer || {};
    webgazer.tracker = webgazer.tracker || {};
    webgazer.util = webgazer.util || {};
    webgazer.params = webgazer.params || {};

    /**
     * Constructor,
     * initialize
     * @constructor
     */
    var jsFeatGaze = function() {};

    webgazer.tracker.jsFeatGaze = jsFeatGaze;
    
    var currentEyes = null;
    
    jsFeatGaze.prototype.getCurrentEyes = function(){
        return currentEyes;
    }

    /**
     * Isolates the two patches that correspond to the user's eyes
     * @param  {Canvas} imageCanvas - canvas corresponding to the webcam stream
     * @param  {Number} width - of imageCanvas
     * @param  {Number} height - of imageCanvas
     * @return {Object} the two eye-patches, first left, then right eye
     */
    jsFeatGaze.prototype.getEyePatches = function(imageCanvas, width, height) {

        if (imageCanvas.width === 0) {
            return null;
        }


        
        var eyes = Detector(imageCanvas).findEyes(imageCanvas);

        //Fit the detected eye in a rectangle
        var leftOriginX = eyes.left.x;
        var leftOriginY = eyes.left.y;
        var leftWidth = eyes.left.width;
        var leftHeight = eyes.left.height;
        var rightOriginX = eyes.right.x;
        var rightOriginY = eyes.right.y;
        var rightWidth = eyes.right.width;
        var rightHeight = eyes.right.height;

        if (leftWidth === 0 || rightWidth === 0) {
            console.log('an eye patch had zero width');
            return null;
        }

        if (leftHeight === 0 || rightHeight === 0) {
            console.log("an eye patch had zero height");
            return null;
        }

        var eyeObjs = {};
        var leftImageData = imageCanvas.getContext('2d').getImageData(leftOriginX, leftOriginY, leftWidth, leftHeight);
        eyeObjs.left = {
            patch: leftImageData,
            imagex: leftOriginX,
            imagey: leftOriginY,
            width: leftWidth,
            height: leftHeight
        };

        var rightImageData = imageCanvas.getContext('2d').getImageData(rightOriginX, rightOriginY, rightWidth, rightHeight);
        eyeObjs.right = {
            patch: rightImageData,
            imagex: rightOriginX,
            imagey: rightOriginY,
            width: rightWidth,
            height: rightHeight
        };

        currentEyes = eyeObjs;

        return eyeObjs;
    };

    /**
     * The object name
     * @type {string}
     */
    jsFeatGaze.prototype.name = 'jsFeatGaze';

    function Detector(canvas) {
            var options = options || new opt();
            var ctx, canvasWidth, canvasHeight;
            var img_u8, work_canvas, work_ctx, ii_sum, ii_sqsum, ii_tilted, edg, ii_canny;
            var classifier = jsfeat.haar.frontalface;
            canvasWidth = canvas.width;
            canvasHeight = canvas.height;

            var max_work_size = 160;

            // var w = canvas.width;
            // var h = canvas.height;
            
                        var scale = Math.min(max_work_size / canvasWidth, max_work_size / canvasWidth);
                        var w = (canvasWidth * scale) | 0;
                        var h = (canvasHeight * scale) | 0;
            
            img_u8 = new jsfeat.matrix_t(w, h, jsfeat.U8_t | jsfeat.C1_t);
            edg = new jsfeat.matrix_t(w, h, jsfeat.U8_t | jsfeat.C1_t);

            work_canvas = document.createElement('canvas');
            work_canvas.width = w;
            work_canvas.height = h;

            ii_sum = new Int32Array((w + 1) * (h + 1));
            ii_sqsum = new Int32Array((w + 1) * (h + 1));
            ii_tilted = new Int32Array((w + 1) * (h + 1));
            ii_canny = new Int32Array((w + 1) * (h + 1));

            var get_identified_facial_data = function(canvas) {

                work_ctx = work_canvas.getContext('2d');
                work_ctx.drawImage(canvas, 0, 0, work_canvas.width, work_canvas.height);
                var imageData = work_ctx.getImageData(0, 0, work_canvas.width, work_canvas.height);

                jsfeat.imgproc.grayscale(imageData.data, work_canvas.width, work_canvas.height, img_u8);

                // possible options
                if (options.equalize_histogram) {
                    jsfeat.imgproc.equalize_histogram(img_u8, img_u8);
                }
                //jsfeat.imgproc.gaussian_blur(img_u8, img_u8, 3);

                jsfeat.imgproc.compute_integral_image(img_u8, ii_sum, ii_sqsum, classifier.tilted ? ii_tilted : null);

                if (options.use_canny) {
                    jsfeat.imgproc.canny(img_u8, edg, 10, 50);
                    jsfeat.imgproc.compute_integral_image(edg, ii_canny, null, null);
                }

                jsfeat.haar.edges_density = options.edges_density;
                var rects = jsfeat.haar.detect_multi_scale(ii_sum, ii_sqsum, ii_tilted, options.use_canny ? ii_canny : null, img_u8.cols, img_u8.rows, classifier, options.scale_factor, options.min_scale);
                rects = jsfeat.haar.group_rectangles(rects, 1);
                
                var scale = canvasWidth / img_u8.cols;
                var scaled_rects = rects.map(function(rect){
                    return {
                        x: rect.x*scale,
                        y:rect.y*scale,
                        width: rect.width*scale,
                        height: rect.height*scale
                    }
                })
                

                return {
                    ctx: ctx,
                    rects: scaled_rects,
                    scale: canvasWidth / img_u8.cols
                }
            };

            function findEyes(imageCanvas) {



                var facial_data = get_identified_facial_data(imageCanvas);
                var face_r = get_most_confident_face_rect(facial_data.rects);
                if (!face_r) {
                    return null;
                }
                var eyes = split_two_eyes(face_r);

                return eyes;


            }

            return {
                findEyes: findEyes
            }


            //----------------------------------------------------------

            function opt() {
                this.min_scale = 2;
                this.scale_factor = 1.15;
                this.use_canny = false;
                this.edges_density = 0.13;
                this.equalize_histogram = true;
            }

            function get_most_confident_face_rect(rects) {
                if (!rects || rects.length == 0) {
                    return null;
                }
                var max = 1;
                var on = rects.length;
                if (on && max) {
                    jsfeat.math.qsort(rects, 0, on - 1, function(a, b) {
                        return (b.confidence < a.confidence);
                    })
                }
                var n = max || on;
                n = Math.min(n, on);
                return rects[0];
            }

            function get_eye_stripe_from_rect(r) {
                var quarter_height = r.height / 4;
                var offsetX = r.x;
                var offsetY = r.y + quarter_height;
                var width = r.width;
                var height = quarter_height;
                return {
                    x: offsetX,
                    y: offsetY,
                    width: width,
                    height: height
                }
            }

            function split_two_eyes(rect) {
                var eye_stripe = get_eye_stripe_from_rect(rect);
                var left = {};
                var right = {};
                var central_distance = eye_stripe.width * 0.11;
                left.x = eye_stripe.x + eye_stripe.width * 0.15;
                left.width = eye_stripe.width * 0.6 / 2;
                left.y = eye_stripe.y + 0;
                left.height = eye_stripe.height;

                right.x = left.x + left.width + central_distance;
                right.width = left.width;
                right.y = left.y;
                right.height = left.height;

                return {
                    left: left,
                    right: right
                }
            }
        }

}(window));
