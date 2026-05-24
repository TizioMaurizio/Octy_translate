import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const HISTORY_LENGTH = 60 // 60 data points = ~2 minutes at 2s interval

interface GpuStats {
  gpu_pct: number
  vram_used_mb: number
  vram_total_mb: number
  temp_c: number
}

interface SystemInfo {
  cpu_pct: number
  ram_used_gb: number
  ram_total_gb: number
  gpu: GpuStats | null
}

interface QueueItem {
  file_name: string
  user_name: string
  queued_at: number
}

interface BotStatus {
  status: 'idle' | 'working' | 'finished' | 'cancelled'
  file_name: string
  elapsed: number
  estimate: string
  duration: number | null
  text: string
  queue: QueueItem[]
  queue_count: number
  system: SystemInfo
  config: {
    model: string
    language: string
    task: string
  }
}

interface LogEntry {
  file_name: string
  file_size_bytes: number
  duration_seconds: number | null
  processing_seconds: number
  model: string
  timestamp: number
}

interface ResourceHistory {
  cpu: number[]
  ram: number[]
  gpu: number[]
  vram: number[]
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString('it-IT')
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    idle: 'badge-idle',
    working: 'badge-working',
    finished: 'badge-finished',
    cancelled: 'badge-cancelled',
  }
  const labels: Record<string, string> = {
    idle: '⏸ In attesa',
    working: '🔄 In corso',
    finished: '✅ Completata',
    cancelled: '🚫 Annullata',
  }
  return <span className={`badge ${colors[status] || ''}`}>{labels[status] || status}</span>
}

function ProgressBar({ elapsed, duration }: { elapsed: number; duration: number | null }) {
  if (!duration || duration <= 0) {
    return <div className="progress-bar"><div className="progress-indeterminate" /></div>
  }
  const pct = Math.min(100, (elapsed / duration) * 100)
  return (
    <div className="progress-bar">
      <div className="progress-fill" style={{ width: `${pct}%` }} />
      <span className="progress-text">{pct.toFixed(1)}%</span>
    </div>
  )
}

function Sparkline({ data, color, max = 100 }: { data: number[]; color: string; max?: number }) {
  const width = 120
  const height = 32
  if (data.length < 2) return <svg width={width} height={height} />

  const effectiveMax = max > 0 ? max : 1
  const points = data.map((v, i) => {
    const x = (i / (HISTORY_LENGTH - 1)) * width
    const y = height - (Math.min(v, effectiveMax) / effectiveMax) * (height - 2) - 1
    return `${x},${y}`
  })
  const pathD = `M${points.join(' L')}`
  // Area fill
  const areaD = `${pathD} L${(data.length - 1) / (HISTORY_LENGTH - 1) * width},${height} L0,${height} Z`

  return (
    <svg width={width} height={height} className="sparkline">
      <path d={areaD} fill={color} fillOpacity="0.15" />
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  )
}

function SystemMonitor({ system, history }: { system: SystemInfo; history: ResourceHistory }) {
  return (
    <div className="system-grid">
      <div className="metric-card">
        <div className="metric-header">
          <div className="metric-label">CPU</div>
          <div className="metric-value">{system.cpu_pct.toFixed(0)}%</div>
        </div>
        <Sparkline data={history.cpu} color="#3b82f6" />
        <div className="metric-bar">
          <div className="metric-fill cpu-fill" style={{ width: `${system.cpu_pct}%` }} />
        </div>
      </div>
      <div className="metric-card">
        <div className="metric-header">
          <div className="metric-label">RAM</div>
          <div className="metric-value">{system.ram_used_gb}/{system.ram_total_gb} GB</div>
        </div>
        <Sparkline data={history.ram} color="#10b981" max={system.ram_total_gb} />
        <div className="metric-bar">
          <div className="metric-fill ram-fill" style={{ width: `${(system.ram_used_gb / system.ram_total_gb) * 100}%` }} />
        </div>
      </div>
      {system.gpu && (
        <>
          <div className="metric-card">
            <div className="metric-header">
              <div className="metric-label">GPU ({system.gpu.temp_c}°C)</div>
              <div className="metric-value">{system.gpu.gpu_pct}%</div>
            </div>
            <Sparkline data={history.gpu} color="#f59e0b" />
            <div className="metric-bar">
              <div className="metric-fill gpu-fill" style={{ width: `${system.gpu.gpu_pct}%` }} />
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-header">
              <div className="metric-label">VRAM</div>
              <div className="metric-value">{(system.gpu.vram_used_mb / 1024).toFixed(1)}/{(system.gpu.vram_total_mb / 1024).toFixed(1)} GB</div>
            </div>
            <Sparkline data={history.vram} color="#8b5cf6" max={system.gpu.vram_total_mb} />
            <div className="metric-bar">
              <div className="metric-fill vram-fill" style={{ width: `${(system.gpu.vram_used_mb / system.gpu.vram_total_mb) * 100}%` }} />
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function App() {
  const [status, setStatus] = useState<BotStatus | null>(null)
  const [log, setLog] = useState<LogEntry[]>([])
  const [connected, setConnected] = useState(false)
  const [tab, setTab] = useState<'live' | 'history'>('live')
  const [resourceHistory, setResourceHistory] = useState<ResourceHistory>({
    cpu: [], ram: [], gpu: [], vram: []
  })
  const wsRef = useRef<WebSocket | null>(null)
  const transcriptRef = useRef<HTMLPreElement>(null)

  const wsUrl = `ws://${window.location.hostname}:3001/api/ws`

  const connect = useCallback(() => {
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => {
      setConnected(false)
      setTimeout(connect, 3000)
    }
    ws.onerror = () => ws.close()
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as BotStatus
        setStatus(data)
        // Accumulate resource history
        setResourceHistory(prev => ({
          cpu: [...prev.cpu, data.system.cpu_pct].slice(-HISTORY_LENGTH),
          ram: [...prev.ram, data.system.ram_used_gb].slice(-HISTORY_LENGTH),
          gpu: [...prev.gpu, data.system.gpu?.gpu_pct ?? 0].slice(-HISTORY_LENGTH),
          vram: [...prev.vram, data.system.gpu?.vram_used_mb ?? 0].slice(-HISTORY_LENGTH),
        }))
      } catch { /* ignore */ }
    }
  }, [wsUrl])

  useEffect(() => {
    connect()
    return () => { wsRef.current?.close() }
  }, [connect])

  useEffect(() => {
    if (tab === 'history') {
      fetch('/api/log')
        .then(r => r.json())
        .then(setLog)
        .catch(() => {})
    }
  }, [tab])

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight
    }
  }, [status?.text])

  const handleCancel = async () => {
    await fetch('/api/cancel', { method: 'POST' })
  }

  if (!status) {
    return (
      <div className="app">
        <div className="connecting">
          <div className="spinner" />
          <p>Connessione al bot...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1>🎙 Octy Transcribe</h1>
          <span className={`connection-dot ${connected ? 'connected' : 'disconnected'}`} />
        </div>
        <div className="header-right">
          <span className="config-badge">Model: {status.config.model}</span>
          <span className="config-badge">Lang: {status.config.language}</span>
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === 'live' ? 'active' : ''} onClick={() => setTab('live')}>
          📡 Live
        </button>
        <button className={tab === 'history' ? 'active' : ''} onClick={() => setTab('history')}>
          📋 Cronologia
        </button>
      </nav>

      {tab === 'live' && (
        <main className="content">
          {/* Status */}
          <section className="card">
            <div className="card-header">
              <h2>Stato Trascrizione</h2>
              <StatusBadge status={status.status} />
            </div>

            {status.status === 'working' && (
              <div className="transcription-info">
                <div className="info-row">
                  <span className="info-label">File:</span>
                  <span className="info-value">{status.file_name}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Trascorso:</span>
                  <span className="info-value">{formatTime(status.elapsed)}</span>
                  <span className="info-label" style={{ marginLeft: '2rem' }}>Stima:</span>
                  <span className="info-value estimate">{status.estimate}</span>
                </div>
                <ProgressBar elapsed={status.elapsed} duration={status.duration} />
                <button className="cancel-btn" onClick={handleCancel}>❌ Annulla</button>
              </div>
            )}

            {status.status === 'finished' && (
              <div className="transcription-info">
                <div className="info-row">
                  <span className="info-label">File:</span>
                  <span className="info-value">{status.file_name}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Completata in:</span>
                  <span className="info-value">{formatTime(status.elapsed)}</span>
                </div>
              </div>
            )}

            {status.status === 'idle' && (
              <p className="idle-text">In attesa di file da trascrivere...</p>
            )}
          </section>

          {/* System Monitor */}
          <section className="card">
            <h2>Risorse Sistema</h2>
            <SystemMonitor system={status.system} history={resourceHistory} />
          </section>

          {/* Transcript */}
          {(status.status === 'working' || status.status === 'finished') && (
            <section className="card transcript-card">
              <h2>Trascrizione Live</h2>
              <pre className="transcript" ref={transcriptRef}>{status.text}</pre>
            </section>
          )}

          {/* Queue */}
          {status.queue_count > 0 && (
            <section className="card">
              <h2>Coda ({status.queue_count})</h2>
              <div className="queue-list">
                {status.queue.map((item, i) => (
                  <div key={i} className="queue-item">
                    <span className="queue-file">{item.file_name}</span>
                    <span className="queue-user">👤 {item.user_name}</span>
                    <span className="queue-time">{formatDate(item.queued_at)}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </main>
      )}

      {tab === 'history' && (
        <main className="content">
          <section className="card">
            <h2>Ultime Trascrizioni</h2>
            {log.length === 0 ? (
              <p className="idle-text">Nessuna trascrizione registrata.</p>
            ) : (
              <div className="history-table">
                <div className="history-header">
                  <span>File</span>
                  <span>Durata audio</span>
                  <span>Tempo elaborazione</span>
                  <span>Modello</span>
                  <span>Data</span>
                </div>
                {[...log].reverse().map((entry, i) => (
                  <div key={i} className="history-row">
                    <span className="history-file">{entry.file_name}</span>
                    <span>{entry.duration_seconds ? formatTime(Math.round(entry.duration_seconds)) : '—'}</span>
                    <span>{formatTime(Math.round(entry.processing_seconds))}</span>
                    <span>{entry.model}</span>
                    <span>{formatDate(entry.timestamp)}</span>
                  </div>
                ))}
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  )
}

export default App
