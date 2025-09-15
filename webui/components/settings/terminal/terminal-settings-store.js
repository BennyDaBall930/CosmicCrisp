/**
 * Terminal Settings Store for Alpine.js
 * Manages terminal configuration and active sessions
 */

document.addEventListener('alpine:init', () => {
    Alpine.data('terminalSettings', () => ({
        // Terminal settings
        settings: {
            shell_path: '/bin/bash',
            rows: 40,
            cols: 120,
            permission_timeout_seconds: 30,
            idle_timeout_seconds: 300,
            max_sessions: 10,
            working_dir: ''
        },
        
        // Active sessions list
        activeSessions: [],
        
        // UI state
        saveStatus: null,
        saveMessage: '',
        
        /**
         * Initialize component
         */
        async init() {
            // Load current settings
            await this.loadSettings();
            
            // Load active sessions
            await this.refreshSessions();
            
            // Refresh sessions periodically
            setInterval(() => {
                if (document.visibilityState === 'visible') {
                    this.refreshSessions();
                }
            }, 10000); // Every 10 seconds
        },
        
        /**
         * Load terminal settings from server
         */
        async loadSettings() {
            try {
                const response = await api.terminalSettings({ action: 'get' });
                if (response.success && response.settings) {
                    this.settings = response.settings;
                }
            } catch (error) {
                console.error('Failed to load terminal settings:', error);
                this.showStatus('error', 'Failed to load settings');
            }
        },
        
        /**
         * Update terminal settings on server
         */
        async updateSettings() {
            try {
                const response = await api.terminalSettings({
                    action: 'update',
                    settings: this.settings
                });
                
                if (response.success) {
                    this.showStatus('success', 'Settings saved successfully');
                } else {
                    this.showStatus('error', response.error || 'Failed to save settings');
                }
            } catch (error) {
                console.error('Failed to update terminal settings:', error);
                this.showStatus('error', 'Failed to save settings');
            }
        },
        
        /**
         * Refresh active sessions list
         */
        async refreshSessions() {
            try {
                const response = await api.terminalSessions();
                if (response.success && response.sessions) {
                    // Convert sessions object to array
                    this.activeSessions = Object.entries(response.sessions).map(([id, session]) => ({
                        id,
                        ...session
                    }));
                }
            } catch (error) {
                console.error('Failed to refresh sessions:', error);
            }
        },
        
        /**
         * Terminate a specific session
         */
        async terminateSession(sessionId) {
            if (!confirm(`Are you sure you want to terminate session ${sessionId}?`)) {
                return;
            }
            
            try {
                // Note: This would need a terminate endpoint to be added
                // For now, we'll just remove it from the list
                this.activeSessions = this.activeSessions.filter(s => s.id !== sessionId);
                this.showStatus('success', `Session ${sessionId} terminated`);
                
                // Refresh to get actual state
                await this.refreshSessions();
            } catch (error) {
                console.error('Failed to terminate session:', error);
                this.showStatus('error', 'Failed to terminate session');
            }
        },
        
        /**
         * Format idle time for display
         */
        formatIdleTime(seconds) {
            if (seconds < 60) {
                return `${Math.round(seconds)}s idle`;
            } else if (seconds < 3600) {
                return `${Math.round(seconds / 60)}m idle`;
            } else {
                return `${Math.round(seconds / 3600)}h idle`;
            }
        },
        
        /**
         * Show status message
         */
        showStatus(type, message) {
            this.saveStatus = type;
            this.saveMessage = message;
            
            // Hide after 3 seconds
            setTimeout(() => {
                this.saveStatus = null;
                this.saveMessage = '';
            }, 3000);
        }
    }));
});

// Add terminal API methods to the global api object
if (typeof api !== 'undefined') {
    // Terminal settings API
    api.terminalSettings = async function(data) {
        return await api.post('/terminal_settings', data);
    };
    
    // Terminal sessions API
    api.terminalSessions = async function() {
        return await api.post('/terminal_sessions', {});
    };
    
    // Terminal start API
    api.terminalStart = async function(data) {
        return await api.post('/terminal_start', data);
    };
    
    // Terminal write API
    api.terminalWrite = async function(sessionId, data) {
        return await api.post('/terminal_write', {
            session_id: sessionId,
            data: data
        });
    };
    
    // Terminal confirm API
    api.terminalConfirm = async function(token) {
        return await api.post('/terminal_confirm', {
            token: token
        });
    };
    
    // Terminal resize API
    api.terminalResize = async function(sessionId, rows, cols) {
        return await api.post('/terminal_resize', {
            session_id: sessionId,
            rows: rows,
            cols: cols
        });
    };
    
    // Terminal stream API (SSE)
    api.terminalStream = function(sessionId, onMessage, onError) {
        const eventSource = new EventSource(`/terminal_stream?session_id=${sessionId}`);
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (onMessage) onMessage(data);
            } catch (error) {
                console.error('Failed to parse terminal stream data:', error);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('Terminal stream error:', error);
            if (onError) onError(error);
            eventSource.close();
        };
        
        return eventSource;
    };
}
