/*
 *
 *    usage example:
        var c = docuement.getElementById("canvas");
        c.width = 100;
        c.height = 100;
        
        //now I want the whole window be divided into 7 * 7 grids
        Calibrator(c, 7, 7);
        
        //event: calibration.finish tells you when user finish calibration
        c.addEventListener("calibration.finish",function(){
            alert("do whatever next");
        });
 *
 */
(function(window){
    
    var makePoints = function(canvas, gridWidth, gridHeight){
        
        if(!canvas.width || (!canvas.height)){
            throw "canvas must have height and width";
        }
        
        var horizontal_grids_count = gridWidth||7;
        var vertical_grids_count = gridHeight||7;
        
        var X_step = parseInt(canvas.width/horizontal_grids_count);
        var Y_step = parseInt(canvas.height/vertical_grids_count);
        
        var makePositions = function(step, bound){
            var positions = [];
            var p = 0;
            
            while(p<bound){
                positions.push(p);
                p+=step;
            }
            
            return positions;
        }
        
        var Xs = makePositions(X_step,canvas.width);
        
        var Ys = makePositions(Y_step,canvas.height);
        
        var flatten = function flatten(arr) {
          return arr.reduce(function (acc, val) {
            return acc.concat(Array.isArray(val) ? flatten(val) : val);
          }, []);
        };        
        
        var positions = Xs.map(function(x){
            return Ys.map(function(y){
                return {
                    x:x,
                    y:y
                }
            });
        });
        
        return {
            x_step: X_step,
            y_step: Y_step,
            points: flatten(positions)
        }
                  
    }
    
    var Calibrator = function(canvas, gridWidth, gridHeight){
        var ctx = canvas.getContext("2d");
        var caliPoints = makePoints(canvas,gridWidth,gridHeight);
        var current_point_index = 0;
        var draw = function(point_index){
            
            if(point_index == (caliPoints.points.length)){
                broadcast(canvas,"calibration.finish");
                return true;
            }
            
            var point = caliPoints.points[point_index];
            
            //avoid to be too narrow a box
            if((canvas.width - (point.x+caliPoints.x_step)) < caliPoints.x_step/2){
                point.x -= caliPoints.x_step/2;
            }
            if((canvas.height- (point.y+caliPoints.y_step))<caliPoints.y_step){
                point.y -= caliPoints.y_step/2;
            }
            
            ctx.fillStyle = "#71D4F5";
            ctx.fillRect(point.x,point.y,caliPoints.x_step, caliPoints.y_step);
            ctx.fillStyle = "white";
            ctx.font = '14pt Calibri';
            ctx.fillText("click this box", point.x+(caliPoints.x_step/gridWidth), point.y+(caliPoints.y_step/gridHeight));
        }
        var validate = function(mousePos,point_index){
            var point = caliPoints.points[point_index];
            var x_bottom = point.x;
            var x_top = point.x+caliPoints.x_step;
            
            var y_bottom = point.y;
            var y_top = point.y+caliPoints.y_step;
            
            if(mousePos.x >= x_bottom && mousePos.x <= x_top &&
                mousePos.y >= y_bottom && mousePos.y <= y_top){
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                current_point_index++;
                draw(current_point_index);
                return true;
            }
            return false;
        }
        
        
        draw(current_point_index);
        canvas.addEventListener("click",function(event){
            //http://www.html5canvastutorials.com/advanced/html5-canvas-mouse-coordinates/
            var getMousePos = function(canvas,evt){
                var rect = canvas.getBoundingClientRect();
                return {
                  x: evt.clientX - rect.left,
                  y: evt.clientY - rect.top
                };                
            }
            var mousePos = getMousePos(canvas, event);
            validate(mousePos,current_point_index);
        });
        
    }
    
    window.Calibrator = Calibrator;
})(window);