(function(window){
	var broadcast = function(elem, name, data){
					        if(data){
					            var event = new CustomEvent(name, {
					                detail:data
					            });
					        }
					        else{
					            var event = new Event(name);
					        }
					        elem.dispatchEvent(event); 
					    };
					    
	var simpleBroadcast = function(name, data){
		var elem = document.querySelector("body");
		broadcast(elem, name, data);
	}
	
	var onSimpleBroadcast = function(name, callback){
		var elem = document.querySelector("body");
		elem.addEventListener(name,callback);
	}
	
	window.broadcast = broadcast;
	window.simpleBroadcast = simpleBroadcast;
	window.onSimpleBroadcast = onSimpleBroadcast;
})(window);