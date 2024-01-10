import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUIStore } from '@/store/ui.store'
import type { Operator } from '@/types'

// Mock operators for development — replace with real auth in production
const MOCK_OPERATORS: Record<string, Operator> = {
  'Cipher@test.com': {
    id: 'opr-001',
    name: 'cipher_ops',
    displayName: 'CIPHER OPS',
    role: 'operator',
    siteId: 'site-001',
  },
  'analyst@cipher.ops': {
    id: 'ana-001',
    name: 'patel_r',
    displayName: 'PATEL_R',
    role: 'analyst',
    siteId: 'site-001',
  },
  'admin@cipher.ops': {
    id: 'adm-001',
    name: 'admin',
    displayName: 'ADMIN',
    role: 'admin',
    siteId: 'site-001',
  },
}

const AUTH_PASSWORD = 'Test123'

export function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { setOperator, addAuditEntry } = useUIStore()
  const navigate = useNavigate()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    // Simulate async auth
    await new Promise((r) => setTimeout(r, 600))

    const operator = MOCK_OPERATORS[email]
    if (operator && password === AUTH_PASSWORD) {
      setOperator(operator)
      addAuditEntry({
        operatorId: operator.id,
        operatorName: operator.displayName,
        action: 'LOGIN',
        detail: `Operator ${operator.displayName} logged in`,
      })
      navigate('/dashboard')
    } else {
      setError('Invalid credentials')
    }

    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-bg-0 flex items-center justify-center">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="font-mono text-2xl text-cloud tracking-[0.12em] mb-1">
            CIPHER
          </div>
          <div className="font-mono text-[10px] text-text-4 tracking-[0.08em] uppercase">
            Operator authentication required
          </div>
        </div>

        {/* Login form */}
        <form
          onSubmit={handleLogin}
          className="bg-bg-1 border border-border rounded-[5px] p-6 flex flex-col gap-4"
        >
          <div>
            <label className="font-mono text-[9px] text-text-3 uppercase tracking-[0.08em] block mb-2">
              Operator email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Cipher@test.com"
              className="w-full bg-bg-2 border border-border rounded-[4px] px-3 py-2
                         font-mono text-[11px] text-text-1 placeholder:text-text-4
                         focus:outline-none focus:border-cloud/40 transition-colors"
              required
            />
          </div>

          <div>
            <label className="font-mono text-[9px] text-text-3 uppercase tracking-[0.08em] block mb-2">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••"
              className="w-full bg-bg-2 border border-border rounded-[4px] px-3 py-2
                         font-mono text-[11px] text-text-1 placeholder:text-text-4
                         focus:outline-none focus:border-cloud/40 transition-colors"
              required
            />
          </div>

          {error && (
            <div className="font-mono text-[10px] text-critical bg-critical/8 border border-critical/20 rounded-[3px] px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 rounded-[4px] font-mono text-[11px] font-bold
                       tracking-[0.06em] uppercase cursor-pointer border transition-all
                       bg-cloud/10 text-cloud border-cloud/25 hover:bg-cloud/20
                       disabled:opacity-40 disabled:cursor-not-allowed mt-1"
          >
            {loading ? 'AUTHENTICATING...' : 'AUTHENTICATE'}
          </button>
        </form>

        {/* Dev hint */}
        {import.meta.env.DEV && (
          <div className="mt-4 bg-bg-2 border border-border/50 rounded-[4px] p-3">
            <div className="font-mono text-[9px] text-text-4 mb-2 tracking-[0.06em]">
              DEV — MOCK CREDENTIALS
            </div>
            <div className="font-mono text-[9px] text-text-3 space-y-1">
              <div>Cipher@test.com / Test123</div>
              <div>analyst@cipher.ops / Test123</div>
              <div>admin@cipher.ops / Test123</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
