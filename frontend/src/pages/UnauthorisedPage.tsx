import { useNavigate } from 'react-router-dom'
import { useUIStore } from '@/store/ui.store'

export function UnauthorisedPage() {
  const navigate = useNavigate()
  const { setOperator } = useUIStore()

  const handleLogout = () => {
    setOperator(null)
    navigate('/login')
  }

  return (
    <div className="min-h-screen bg-bg-0 flex items-center justify-center">
      <div className="text-center">
        <div className="font-mono text-[10px] text-text-4 tracking-[0.08em] uppercase mb-3">
          SENTINEL/OPS
        </div>
        <div className="font-mono text-xl text-critical mb-2 tracking-[0.06em]">
          ACCESS DENIED
        </div>
        <div className="font-mono text-[11px] text-text-3 mb-6">
          Your role does not have permission to access this page.
        </div>
        <button
          onClick={handleLogout}
          className="font-mono text-[10px] px-4 py-2 border border-border
                     text-text-3 rounded-[4px] hover:text-text-2 hover:border-border-2
                     transition-colors cursor-pointer bg-transparent"
        >
          RETURN TO LOGIN
        </button>
      </div>
    </div>
  )
}
