// Global variables
let data = {};
let currentZoom = 1;
let minTime = Infinity;
let maxTime = -Infinity;

// Optimization variables
let virtualizedLanes = [];
let visibleLanes = new Set();
let renderAnimationFrame = null;

// UI state variables
let isResizing = false;
let currentResizeElement = null;
let startX = 0;
let startWidth = 0;
let hiddenLanes = new Set();

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
    
    // Remove loading class to enable scrolling
    const timelineWrapper = document.querySelector('.timeline-wrapper');
    if (timelineWrapper) {
        timelineWrapper.classList.remove('loading');
    }
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
 * Update all event bars based on current zoom level with performance optimizations
 */
function updateEventBars() {
    // Use requestAnimationFrame for smooth updates
    if (renderAnimationFrame) {
        cancelAnimationFrame(renderAnimationFrame);
    }
    
    renderAnimationFrame = requestAnimationFrame(() => {
        const duration = maxTime - minTime;
        const baseWidth = 1000;
        const scaledWidth = baseWidth * currentZoom;
        
        // Update the timeline content container
        const timelineContent = document.getElementById('timeline-app');
        timelineContent.style.minWidth = `${scaledWidth + 250}px`;
        
        // Update all event bar wrappers
        const eventBarWrappers = document.querySelectorAll('.event-bar-wrapper');
        eventBarWrappers.forEach(wrapper => {
            wrapper.style.width = `${scaledWidth}px`;
            wrapper.style.minWidth = `${scaledWidth}px`;
        });

        // Batch update visible event bars
        const eventBars = document.querySelectorAll('.event-bar');
        const fragment = document.createDocumentFragment();
        const barsToUpdate = [];
        
        eventBars.forEach(bar => {
            const start = parseFloat(bar.dataset.start);
            const end = parseFloat(bar.dataset.end);
            
            // Calculate positions
            const leftPx = ((start - minTime) / duration) * scaledWidth;
            const widthPx = Math.max(((end - start) / duration) * scaledWidth, 0.5);
            
            barsToUpdate.push({ bar, leftPx, widthPx });
        });
        
        // Apply updates in batches to minimize reflow
        barsToUpdate.forEach(({ bar, leftPx, widthPx }) => {
            bar.style.left = `${leftPx}px`;
            bar.style.width = `${widthPx}px`;
        });
        
        renderAnimationFrame = null;
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
 * Debounce function to limit function calls
 * @param {Function} func - Function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} - Debounced function
 */
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

/**
 * Throttle function to limit function calls
 * @param {Function} func - Function to throttle
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} - Throttled function
 */
function throttle(func, delay) {
    let timeoutId;
    let lastExecTime = 0;
    return function(...args) {
        const currentTime = Date.now();
        if (currentTime - lastExecTime > delay) {
            func.apply(this, args);
            lastExecTime = currentTime;
        } else {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                func.apply(this, args);
                lastExecTime = Date.now();
            }, delay - (currentTime - lastExecTime));
        }
    };
}

/**
 * Create a virtualized event lane for better performance
 * @param {Object} laneData - Lane configuration
 * @returns {HTMLElement} - The created lane element
 */
function createVirtualizedLane(laneData) {
    const { eventName, events, eventColor, duration, profileName } = laneData;
    
    const eventLane = document.createElement('div');
    eventLane.className = 'event-lane';
    eventLane.dataset.profileName = profileName;
    eventLane.dataset.eventName = eventName;
    
    const eventNameLabel = document.createElement('div');
    eventNameLabel.className = 'event-name';
    eventNameLabel.textContent = eventName;
    eventNameLabel.style.borderLeft = `4px solid ${eventColor}`;
    
    // Add title attribute for tooltip if name is long (adjusted for 250px width)
    if (eventName.length > 30) {
        eventNameLabel.setAttribute('title', eventName);
    }
    
    // Add click handler for hide/move functionality
    eventNameLabel.addEventListener('click', (e) => {
        handleEventNameClick(eventLane, eventName, e);
    });
    
    // Add resize functionality
    eventNameLabel.addEventListener('mousedown', (e) => {
        handleResizeStart(e, eventNameLabel);
    });
    
    eventLane.appendChild(eventNameLabel);
    
    const eventBarContainer = document.createElement('div');
    eventBarContainer.className = 'event-bar-container';

    const eventBarWrapper = document.createElement('div');
    eventBarWrapper.className = 'event-bar-wrapper';
    eventBarWrapper.dataset.eventCount = events.length;

    // Only render events that are visible or significant
    const significantEvents = filterSignificantEvents(events, duration);
    
    significantEvents.forEach(event => {
        const bar = createEventBar(event, eventColor, eventName, duration);
        eventBarWrapper.appendChild(bar);
    });
    
    eventBarContainer.appendChild(eventBarWrapper);
    eventLane.appendChild(eventBarContainer);
    
    return eventLane;
}

/**
 * Filter events to only show significant ones at current zoom level
 * @param {Array} events - Array of events
 * @param {number} duration - Total timeline duration
 * @returns {Array} - Filtered events
 */
function filterSignificantEvents(events, duration) {
    const baseWidth = 1000;
    const scaledWidth = baseWidth * currentZoom;
    const minVisibleWidth = 1; // Minimum width in pixels to be visible
    const minDurationToShow = (minVisibleWidth / scaledWidth) * duration;
    
    // If zoomed out significantly, merge nearby events
    if (currentZoom < 0.5) {
        return mergeNearbyEvents(events, duration);
    }
    
    // Filter out events that are too small to see
    return events.filter(event => {
        const eventDuration = event[1] - event[0];
        return eventDuration >= minDurationToShow;
    });
}

/**
 * Merge nearby events when zoomed out for better performance
 * @param {Array} events - Array of events
 * @param {number} duration - Total timeline duration
 * @returns {Array} - Merged events
 */
function mergeNearbyEvents(events, duration) {
    if (events.length === 0) return events;
    
    const baseWidth = 1000;
    const scaledWidth = baseWidth * currentZoom;
    const mergeThreshold = (5 / scaledWidth) * duration; // 5 pixels
    
    const merged = [];
    let currentGroup = [events[0]];
    
    for (let i = 1; i < events.length; i++) {
        const prevEvent = currentGroup[currentGroup.length - 1];
        const currentEvent = events[i];
        
        if (currentEvent[0] - prevEvent[1] <= mergeThreshold) {
            currentGroup.push(currentEvent);
        } else {
            // Merge the current group
            if (currentGroup.length === 1) {
                merged.push(currentGroup[0]);
            } else {
                const mergedEvent = [
                    currentGroup[0][0], // Start of first event
                    currentGroup[currentGroup.length - 1][1], // End of last event
                    { merged: currentGroup.length } // Metadata indicating merge
                ];
                merged.push(mergedEvent);
            }
            currentGroup = [currentEvent];
        }
    }
    
    // Handle the last group
    if (currentGroup.length === 1) {
        merged.push(currentGroup[0]);
    } else {
        const mergedEvent = [
            currentGroup[0][0],
            currentGroup[currentGroup.length - 1][1],
            { merged: currentGroup.length }
        ];
        merged.push(mergedEvent);
    }
    
    return merged;
}

/**
 * Create an individual event bar
 * @param {Array} event - Event data
 * @param {string} eventColor - Color for the event
 * @param {string} eventName - Name of the event
 * @param {number} duration - Total timeline duration
 * @returns {HTMLElement} - The created event bar
 */
function createEventBar(event, eventColor, eventName, duration) {
    const start = event[0];
    const end = event[1];
    const metadata = event.length > 2 ? event[2] : null;
    
    const bar = document.createElement('div');
    bar.className = 'event-bar';
    bar.style.backgroundColor = eventColor;
    bar.dataset.start = start;
    bar.dataset.end = end;
    
    // Initial positioning
    const baseWidth = 1000;
    const leftPx = ((start - minTime) / duration) * baseWidth;
    const widthPx = Math.max(((end - start) / duration) * baseWidth, 0.5);
    bar.style.left = `${leftPx}px`;
    bar.style.width = `${widthPx}px`;
    
    // Use throttled tooltip handlers
    const throttledShowTooltip = throttle((e) => {
        showTooltip(e, eventName, start, end, eventColor, metadata);
    }, 16); // ~60fps
    
    bar.addEventListener('mouseenter', throttledShowTooltip);
    bar.addEventListener('mouseleave', hideTooltip);
    bar.addEventListener('mousemove', throttledShowTooltip);
    
    return bar;
}

/**
 * Create the timeline visualization with performance optimizations
 * @param {string} containerId - ID of the container element
 */
function createTimeline(containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = ''; // Clear existing content

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
    console.log(`Timeline duration: ${duration.toFixed(3)}s (${minTime.toFixed(3)}s - ${maxTime.toFixed(3)}s)`);

    // Create a document fragment for batch DOM operations
    const fragment = document.createDocumentFragment();

    for (const profileName in data) {
        const profileContainer = document.createElement('div');
        profileContainer.className = 'timeline-container';
        
        const profileTitle = document.createElement('h3');
        profileTitle.textContent = `${profileName} (${Object.keys(data[profileName]).length} events)`;
        profileContainer.appendChild(profileTitle);
        
        const eventData = data[profileName];
        const sortedEventNames = Object.keys(eventData).sort();

        // Create lanes using virtualization
        sortedEventNames.forEach((eventName, eventIndex) => {
            const eventColor = getEventColor(eventName, eventIndex);
            const events = eventData[eventName];
            
            const laneData = {
                eventName,
                events,
                eventColor,
                duration,
                profileName
            };
            
            const lane = createVirtualizedLane(laneData);
            virtualizedLanes.push({
                element: lane,
                data: laneData,
                visible: true
            });
            
            profileContainer.appendChild(lane);
        });
        
        fragment.appendChild(profileContainer);
    }
    
    // Batch DOM update
    container.appendChild(fragment);
    
    // Add optimized wheel event listener
    const timelineWrapper = document.querySelector('.timeline-wrapper');
    const throttledWheelZoom = throttle(handleWheelZoom, 16);
    timelineWrapper.addEventListener('wheel', throttledWheelZoom, { passive: false });
    
    // Add intersection observer for viewport culling
    setupViewportCulling();
    
    console.log(`Timeline created with ${virtualizedLanes.length} lanes, ${getTotalEventCount()} total events`);
}

/**
 * Get total event count across all lanes
 * @returns {number} Total number of events
 */
function getTotalEventCount() {
    return virtualizedLanes.reduce((total, lane) => {
        return total + lane.data.events.length;
    }, 0);
}

/**
 * Setup viewport culling to hide lanes that are not visible
 */
function setupViewportCulling() {
    if (!window.IntersectionObserver) {
        console.log('IntersectionObserver not supported, skipping viewport culling');
        return;
    }
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const lane = virtualizedLanes.find(l => l.element === entry.target);
            if (lane) {
                const wasVisible = lane.visible;
                lane.visible = entry.isIntersecting;
                
                // Only update if visibility changed
                if (wasVisible !== lane.visible) {
                    const eventBars = lane.element.querySelectorAll('.event-bar');
                    eventBars.forEach(bar => {
                        bar.style.display = lane.visible ? 'block' : 'none';
                    });
                }
            }
        });
    }, {
        root: document.querySelector('.timeline-wrapper'),
        rootMargin: '50px 0px', // Load lanes 50px before they come into view
        threshold: 0
    });
    
    // Observe all lanes
    virtualizedLanes.forEach(lane => {
        observer.observe(lane.element);
    });
}

/**
 * Handle event name click for hide/show functionality
 * @param {HTMLElement} eventLane - The event lane element
 * @param {string} eventName - Name of the event
 * @param {Event} event - The click event
 */
function handleEventNameClick(eventLane, eventName, event) {
    // Check if this is a resize action (near right edge)
    const rect = event.target.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const isNearRightEdge = clickX > rect.width - 10;
    
    if (isNearRightEdge) {
        return; // Don't handle click if it's a resize action
    }
    
    const laneId = `${eventLane.dataset.profileName}_${eventName}`;
    
    // Simple toggle between normal and hidden
    if (eventLane.classList.contains('hidden')) {
        // Restore from hidden to normal
        hiddenLanes.delete(laneId);
        eventLane.classList.remove('hidden');
        console.log(`Restored lane: ${eventName}`);
    } else {
        // Hide the lane
        hiddenLanes.add(laneId);
        eventLane.classList.add('hidden');
        console.log(`Hidden lane: ${eventName}`);
    }
    
    updateLaneStatusDisplay();
}

/**
 * Handle resize start
 * @param {Event} event - The mousedown event
 * @param {HTMLElement} element - The event name element
 */
function handleResizeStart(event, element) {
    const rect = element.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const isNearRightEdge = clickX > rect.width - 10;
    
    if (!isNearRightEdge) {
        return; // Only start resize if near right edge
    }
    
    event.preventDefault();
    isResizing = true;
    currentResizeElement = element;
    startX = event.clientX;
    startWidth = element.offsetWidth;
    
    document.addEventListener('mousemove', handleResize);
    document.addEventListener('mouseup', handleResizeEnd);
    document.body.style.cursor = 'col-resize';
}

/**
 * Handle resize during drag
 * @param {Event} event - The mousemove event
 */
function handleResize(event) {
    if (!isResizing || !currentResizeElement) return;
    
    const deltaX = event.clientX - startX;
    const newWidth = Math.max(150, Math.min(600, startWidth + deltaX));
    
    // Update all event name elements to maintain alignment
    const allEventNames = document.querySelectorAll('.event-name');
    allEventNames.forEach(el => {
        el.style.width = `${newWidth}px`;
    });
}

/**
 * Handle resize end
 */
function handleResizeEnd() {
    isResizing = false;
    currentResizeElement = null;
    document.removeEventListener('mousemove', handleResize);
    document.removeEventListener('mouseup', handleResizeEnd);
    document.body.style.cursor = '';
}

/**
 * Update lane status display in the controls
 */
function updateLaneStatusDisplay() {
    const hiddenCount = hiddenLanes.size;
    
    // Add or update status display
    let statusElement = document.getElementById('lane-status');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = 'lane-status';
        statusElement.style.cssText = `
            margin-left: 15px;
            padding: 4px 8px;
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            font-size: 0.8em;
            color: #856404;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        `;
        document.querySelector('.zoom-controls').appendChild(statusElement);
    }
    
    if (hiddenCount > 0) {
        statusElement.innerHTML = `
            <span>Hidden lanes: ${hiddenCount}</span>
            <button onclick="restoreAllLanes()" style="
                background: #856404;
                color: white;
                border: none;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.9em;
                cursor: pointer;
            ">Restore All</button>
        `;
        statusElement.style.display = 'inline-flex';
    } else {
        statusElement.style.display = 'none';
    }
}

/**
 * Restore all hidden lanes to their original positions
 */
function restoreAllLanes() {
    // Get all lanes
    const allLanes = document.querySelectorAll('.event-lane');
    
    allLanes.forEach(lane => {
        lane.classList.remove('hidden');
    });
    
    // Clear the state set
    hiddenLanes.clear();
    
    updateLaneStatusDisplay();
}

// Make restoreAllLanes available globally for the button onclick
window.restoreAllLanes = restoreAllLanes;
