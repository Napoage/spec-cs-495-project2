// Fetch JSON and initialize sliders
async function initializeSliders() {
try {
    // Fetch current_points from app.py. current_points are the coords of the trapezoid
    // even if the user hasn't saved it to config.json. These current_points are stored
    // until the app is restarted.
    const response = await fetch('/get_current_points')
    const currentPoints = await response.json();
    console.log("currentPoints in javascript: ", currentPoints)


    // Get slider elements
    const sliders = {
        top_left_x: document.getElementById('top_left_x'),
        top_left_y: document.getElementById('top_left_y'),
        top_right_x: document.getElementById('top_right_x'),
        bottom_left_y: document.getElementById('bottom_left_y')
    };

    // Map points to sliders based on indices
    sliders.top_left_x.value = currentPoints[1][0]; // Top Left X
    sliders.top_left_y.value = currentPoints[1][1]; // Top Left Y
    sliders.top_right_x.value = currentPoints[0][0]; // Top Right X
    sliders.bottom_left_y.value = currentPoints[2][1]; // Bottom Left Y

    // Add input event listeners to sliders
    Object.keys(sliders).forEach(sliderId => {
        sliders[sliderId].addEventListener('input', () => {
            // Log updated values if needed
            console.log(`${sliderId} updated to ${sliders[sliderId].value}`);
        });
    });

} catch (error) {
    console.error('Error initializing sliders:', error);
}
}
// Initialize sliders on page load
initializeSliders();