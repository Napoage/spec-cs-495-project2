function slidePoint(point, value) {
    console.log(`Adjusting ${point} to value: ${value}`);
    
    /*const response = await fetch('/get_current_points')
    const currentPoints = await response.json();
    console.log("currentPoints in javascript: ", currentPoints)*/
    
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/slide_point?point=${point}&value=${value}`, true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            console.log(`Moved ${point} ${value}`); 
            console.log('Slider values:', {
            topLeftX: topLeftXSlider.value,
            topRightX: topRightXSlider.value,
            topLeftY: topLeftYSlider.value,
            bottomLeftY: bottomLeftYSlider.value
        });
            // Refresh the video feed to show updated trapezoid
            document.getElementById('video_feed').src = '/process_trapezoid?' + new Date().getTime(); 
        } else {
            console.error('Error moving point:', xhr.responseText);
        }
    };
    xhr.onerror = function () {
        console.error('Request failed');
    };
    xhr.send();

}

//Define these variables globally    
const topLeftXSlider = document.getElementById('top_left_x');
const topRightXSlider = document.getElementById('top_right_x');
const topLeftYSlider = document.getElementById('top_left_y');
const bottomLeftYSlider = document.getElementById('bottom_left_y');

function updateMaxMinSlider() {
    console.log("in updatemaxminslider");
 
    // Get the current value of the sliders
    const topLeftXValue = parseInt(topLeftXSlider.value, 10);
    const topLeftYValue = parseInt(topLeftYSlider.value, 10);
    const topRightXValue = parseInt(topRightXSlider.value, 10);
    const bottomLeftYValue = parseInt(bottomLeftYSlider.value, 10);

    console.log("top_left_x = ", topLeftXValue);
    console.log("top_left_y = ", topLeftYValue);
    console.log("top_right_x = ", topRightXValue);
    console.log("bottom_left_y = ", bottomLeftYValue);

    // Dynamically set the max value of the sliders
    const topLeftXMax = topRightXValue - 10; //Make it 10 pixels shy of the top right x value. 
    topLeftXSlider.max = topLeftXMax;
    console.log("topLeftXMax = ", topLeftXMax); 
    if (parseInt(topLeftXSlider.value, 10) > topLeftXMax) {
        topLeftXSlider.value = topLeftXMax;
    }

    const topRightXMin = topLeftXValue + 10; 
    topRightXSlider.min = topRightXMin;
    console.log("topRightXMin = ", topRightXMin);
    if (parseInt(topRightXSlider.value, 10) < topRightXMin) {
        topRightXSlider.value = topRightXMin;
    }

    const bottomLeftYMin = topLeftYValue + 10; 
    bottomLeftYSlider.min = bottomLeftYMin;
    if (parseInt(bottomLeftYSlider.value, 10) < bottomLeftYMin) {
        bottomLeftYSlider.value = bottomLeftYMin;
    }      

    const topLeftYMax = bottomLeftYValue - 10; 
    topLeftYSlider.max = topLeftYMax;
    if (parseInt(topLeftYSlider.value, 10) > topLeftYMax) {
        topLeftYSlider.value = topLeftYMax;
    } 

}
function logSliderValues() {
    console.log('Slider values:', {
    topLeftX: topLeftXSlider.value,
    topRightX: topRightXSlider.value,
    topLeftY: topLeftYSlider.value,
    bottomLeftY: bottomLeftYSlider.value
});
}

// Add event listeners for real-time constraint updates
[topLeftXSlider, topRightXSlider, topLeftYSlider, bottomLeftYSlider].forEach(slider => {slider.addEventListener('input', () => {logSliderValues(); updateMaxMinSlider(); } )});

function savePoints() {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/save_points', true);
    xhr.onload = function () {
        if (xhr.status === 200) {
            alert('SPEC: Points saved successfully!');
        } else {
            alert('Error saving points: ' + xhr.responseText);
        }
    };
    xhr.send();
}
function transformed() {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '/transformed_image', true); // Request the transformed image from the server
    xhr.onload = function () {
        if (xhr.status === 200) {
            // Generate a unique URL by appending a timestamp to prevent caching
            const transformedImage = document.getElementById('transformed-image');
            const uniqueUrl = '/static/mask/captured_frame.jpg?' + new Date().getTime(); // Add timestamp
            transformedImage.src = uniqueUrl;
            transformedImage.style.display = 'block'; // Show the image
        } else {
            console.error('Error fetching transformed image:', xhr.responseText);
        }
    };
    xhr.onerror = function () {
        console.error('Request failed');
    };
    xhr.send();
}