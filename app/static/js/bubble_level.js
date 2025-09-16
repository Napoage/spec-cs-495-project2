
    async function read_IMU_for_level() {
      try {
        const response = await fetch('/read_IMU_for_level');
        const IMUReading = await response.json();
        // console.log("IMU Reading:", IMUReading);        
        return IMUReading; //convert to degrees 
      } catch (error) {
        console.error("Error fetching IMU data:", error);
        // Default values in case of an error
        return [0, 0, 0]; // [pitch, roll, yaw]
      }
    }

    async function initializeBubbleLevel() {
      const vial = document.querySelector('.vial');
      const bubble = document.querySelector('.bubble');
      const rollDisplay = document.getElementById('roll-angle');
      //const pitchDisplay = document.getElementById('pitch-angle');

      // Maximum horizontal displacement of the bubble within the vial
      const maxDisplacement = (vial.offsetWidth - bubble.offsetWidth) / 2;
    
      //function updateBubbleAndRollDisplay(pitch, roll) {
      function updateBubbleAndRollDisplay(roll) {  
        // Clamp the roll angle to [-90°, 90°] for visualization purposes

        const clampedRoll = Math.max(-90, Math.min(90, roll));
        //const clampedPitch = Math.max(-180, Math.min(180, pitch));

        // Update the horizontal position of the bubble based on roll
        const x = (clampedRoll / 90) * maxDisplacement; // Map roll to vial width
        const bubbleWidth = 60; //width in pixels from css
        const xOffset = x - bubbleWidth / 2; // Adjust for the bubble's width
        bubble.style.transform = `translate(${xOffset}px, -50%)`;

        // Update the roll text display
        rollDisplay.textContent = clampedRoll.toFixed(0); // Show roll with 0 decimal places
        //pitchDisplay.textContent = clampedPitch.toFixed(2);
        //pitchDisplay.textContent = pitch.toFixed(0);
      }

      // Initial reading
      // Multiply by 57.3 to convert radians to degrees
      const initialReading = await read_IMU_for_level();
      //console.log("pitch deg = ", initialReading[0]*57.3);
      //updateBubbleAndRollDisplay(initialReading[0]*57.3, initialReading[1]*57.3);// Roll is the second value
      updateBubbleAndRollDisplay(initialReading[1]*57.3); 

      // Periodic updates
      setInterval(async () => {
        const updatedReading = await read_IMU_for_level();
        //updateBubbleAndRollDisplay(updatedReading[0]*57.3, updatedReading[1]*57.3);
        updateBubbleAndRollDisplay(updatedReading[1]*57.3);
      }, 300); // Update every 0.3 second
    }

    document.addEventListener("DOMContentLoaded", initializeBubbleLevel);
