// Global variables
let data = {};
let currentZoom = 1;
let minTime = Infinity;
let maxTime = -Infinity;

// TensorBoard-inspired color palette
const colorPalette = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
    '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
    '#FF6348', '#2ED573', '#3742FA', '#F368E0', '#FFA502',
    '#FF3838', '#1DD1A1', '#5352ED', '#FF9FF3', '#FFC048',
    '#FF4757', '#2ED573', '#3742FA', '#FF6B9D', '#C44569',
    '#F8B500', '#78E08F', '#4834D4', '#FF6B9D', '#6C5CE7'
];

/**
 * Initialize the timeline with data
 * @param {Object} timelineData - The timeline data to display
 */
function initializeTimeline(timelineData) {
    data = timelineData;
    createTimeline('timeline-app');
}

/**
 * Get a consistent color for an event based on its name and index
 * @param {string} eventName - The name of the event
 * @param {number} eventIndex - The index of the event
 * @returns {string} - A color from the palette
 */
function getEventColor(eventName, eventIndex) {
    // Create a hash from the event name for consistent coloring
    let hash = 0;
    for (let i = 0; i < eventName.length; i++) {
        const char = eventName.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Use a combination of hash and index for better distribution
    const colorIndex = (Math.abs(hash) + eventIndex) % colorPalette.length;
    return colorPalette[colorIndex];
}

/**
 * Format duration for display
 * @param {number} duration - Duration in seconds
 * @returns {string} - Formatted duration string
 */
function formatDuration(duration) {
    if (duration < 0.001) {
        return `${(duration * 1000000).toFixed(2)}μs`; // microseconds
    } else if (duration < 1) {
        return `${(duration * 1000).toFixed(2)}ms`; // milliseconds
    } else {
        return `${duration.toFixed(3)}s`; // seconds
    }
}

/**
 * Show tooltip with event information
 * @param {Event} event - Mouse event
 * @param {string} eventName - Name of the event
 * @param {number} start - Start time
 * @param {number} end - End time
 * @param {string} color - Event color
 * @param {Object|null} metadata - Optional metadata object
 */
function showTooltip(event, eventName, start, end, color, metadata = null) {
    const tooltip = document.getElementById('tooltip');
    const duration = end - start;
    
    let metadataHtml = '';
    if (metadata && typeof metadata === 'object') {
        metadataHtml = '<div class="metadata-section">';
        for (const [key, value] of Object.entries(metadata)) {
            metadataHtml += `<div class="metadata-item">${key}: ${value}</div>`;
        }
        metadataHtml += '</div>';
    }
    
    tooltip.innerHTML = `
        <strong>${eventName}</strong><br>
        <div style="color: #a0c4ff;">Start: ${start.toFixed(6)}s</div>
        <div style="color: #a0c4ff;">End: ${end.toFixed(6)}s</div>
        <div style="color: ${color};">Duration: ${formatDuration(duration)}</div>
        ${metadataHtml}
    `;
    tooltip.style.display = 'block';
    tooltip.style.left = event.pageX + 15 + 'px';
    tooltip.style.top = event.pageY - 10 + 'px';
}

/**
 * Hide the tooltip
 */
function hideTooltip() {
    document.getElementById('tooltip').style.display = 'none';
}

/**
 * Update the zoom level display and event bars
 */
function updateZoomLevel() {
    document.getElementById('zoom-level').textContent = `Zoom: ${Math.round(currentZoom * 100)}%`;
    updateEventBars();
}

/**
 * Update all event bars based on current zoom level
 */
function updateEventBars() {
    const duration = maxTime - minTime;
    const baseWidth = 1000; // Base width for timeline
    const scaledWidth = baseWidth * currentZoom;
    
    // Update the timeline content container
    const timelineContent = document.getElementById('timeline-app');
    timelineContent.style.minWidth = `${scaledWidth + 250}px`; // Add extra space for labels
    
    // Update all event bar wrappers
    const eventBarWrappers = document.querySelectorAll('.event-bar-wrapper');
    eventBarWrappers.forEach(wrapper => {
        wrapper.style.width = `${scaledWidth}px`;
        wrapper.style.minWidth = `${scaledWidth}px`;
    });

    // Update all event bars positioning and sizing using absolute positions
    const eventBars = document.querySelectorAll('.event-bar');
    eventBars.forEach(bar => {
        const start = parseFloat(bar.dataset.start);
        const end = parseFloat(bar.dataset.end);
        
        // Calculate absolute positions within the scaled container
        const leftPx = ((start - minTime) / duration) * scaledWidth;
        const widthPx = ((end - start) / duration) * scaledWidth;
        
        bar.style.left = `${leftPx}px`;
        bar.style.width = `${Math.max(widthPx, 1)}px`; // Ensure minimum width
    });
}

/**
 * Zoom in function (called by button)
 */
function zoomIn() {
    currentZoom = Math.min(currentZoom * 1.5, 10);
    updateZoomLevel();
}

/**
 * Zoom out function (called by button)
 */
function zoomOut() {
    currentZoom = Math.max(currentZoom / 1.5, 0.1);
    updateZoomLevel();
}

/**
 * Reset zoom to 100%
 */
function resetZoom() {
    currentZoom = 1;
    updateZoomLevel();
}

/**
 * Handle wheel zoom with shift key
 * @param {WheelEvent} event - The wheel event
 */
function handleWheelZoom(event) {
    // Check if shift key is held down
    if (event.shiftKey) {
        event.preventDefault(); // Prevent default scrolling
        
        // Get the timeline wrapper element
        const timelineWrapper = document.querySelector('.timeline-wrapper');
        const rect = timelineWrapper.getBoundingClientRect();
        
        // Calculate mouse position relative to the timeline wrapper
        const mouseX = event.clientX - rect.left;
        const scrollLeft = timelineWrapper.scrollLeft;
        
        // Calculate the mouse position in the timeline content
        const mousePositionInContent = mouseX + scrollLeft;
        
        // Store the old zoom level
        const oldZoom = currentZoom;
        
        // Determine zoom direction based on wheel delta
        const delta = event.deltaY || event.wheelDelta;
        
        if (delta < 0) {
            // Scrolling up - zoom in
            currentZoom = Math.min(currentZoom * 1.05, 10);
        } else {
            // Scrolling down - zoom out
            currentZoom = Math.max(currentZoom / 1.05, 0.1);
        }
        
        // Update the zoom level display and bars
        updateZoomLevel();
        
        // Calculate the new position after zoom
        const zoomRatio = currentZoom / oldZoom;
        const newMousePositionInContent = mousePositionInContent * zoomRatio;
        
        // Calculate the new scroll position to keep the mouse position fixed
        const newScrollLeft = newMousePositionInContent - mouseX;
        
        // Apply the new scroll position
        timelineWrapper.scrollLeft = Math.max(0, newScrollLeft);
    }
}

/**
 * Create the timeline visualization
 * @param {string} containerId - ID of the container element
 */
function createTimeline(containerId) {
    const container = document.getElementById(containerId);

    // Find global min and max times to set the timeline scale
    for (const profileName in data) {
        for (const eventName in data[profileName]) {
            data[profileName][eventName].forEach(event => {
                minTime = Math.min(minTime, event[0]);
                maxTime = Math.max(maxTime, event[1]);
            });
        }
    }
    
    const duration = maxTime - minTime;

    for (const profileName in data) {
        const profileContainer = document.createElement('div');
        profileContainer.className = 'timeline-container';
        
        const profileTitle = document.createElement('h3');
        profileTitle.textContent = profileName;
        profileContainer.appendChild(profileTitle);
        
        const eventData = data[profileName];
        const sortedEventNames = Object.keys(eventData).sort();

        sortedEventNames.forEach((eventName, eventIndex) => {
            const eventColor = getEventColor(eventName, eventIndex);
            
            const eventLane = document.createElement('div');
            eventLane.className = 'event-lane';
            
            const eventNameLabel = document.createElement('div');
            eventNameLabel.className = 'event-name';
            eventNameLabel.textContent = eventName;
            // Add a colored indicator next to the event name
            eventNameLabel.style.borderLeft = `4px solid ${eventColor}`;
            eventLane.appendChild(eventNameLabel);
            
            const eventBarContainer = document.createElement('div');
            eventBarContainer.className = 'event-bar-container';

            const eventBarWrapper = document.createElement('div');
            eventBarWrapper.className = 'event-bar-wrapper';

            eventData[eventName].forEach(event => {
                const start = event[0];
                const end = event[1];
                const metadata = event.length > 2 ? event[2] : null;
                const bar = document.createElement('div');
                bar.className = 'event-bar';
                
                // Apply the color to this event bar
                bar.style.backgroundColor = eventColor;
                
                // Store start and end times as data attributes for zoom calculations
                bar.dataset.start = start;
                bar.dataset.end = end;
                
                // Initial positioning using pixels instead of percentages
                const baseWidth = 1000;
                const leftPx = ((start - minTime) / duration) * baseWidth;
                const widthPx = ((end - start) / duration) * baseWidth;
                bar.style.left = `${leftPx}px`;
                bar.style.width = `${widthPx}px`;
                
                // Add tooltip functionality with color and metadata
                bar.addEventListener('mouseenter', (e) => {
                    showTooltip(e, eventName, start, end, eventColor, metadata);
                });
                bar.addEventListener('mouseleave', hideTooltip);
                bar.addEventListener('mousemove', (e) => {
                    showTooltip(e, eventName, start, end, eventColor, metadata);
                });
                
                eventBarWrapper.appendChild(bar);
            });
            
            eventBarContainer.appendChild(eventBarWrapper);
            eventLane.appendChild(eventBarContainer);
            profileContainer.appendChild(eventLane);
        });
        
        container.appendChild(profileContainer);
    }
    
    // Add wheel event listener to the timeline wrapper for zoom functionality
    const timelineWrapper = document.querySelector('.timeline-wrapper');
    timelineWrapper.addEventListener('wheel', handleWheelZoom);
}
