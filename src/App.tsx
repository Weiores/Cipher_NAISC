import { Routes, Route, Navigate } from 'react-router-dom'
import { DashboardPage } from '@/pages/DashboardPage'
import { LoginPage } from '@/pages/LoginPage'
import { UnauthorisedPage } from '@/pages/UnauthorisedPage'
import { ProtectedRoute } from '@/components/layout/ProtectedRoute'
import { useUIStore } from '@/store/ui.store'

export default function App() {
  const operator = useUIStore((s) => s.operator)

  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/unauthorised" element={<UnauthorisedPage />} />

      {/* Protected — operator + admin + analyst */}
      <Route element={<ProtectedRoute allowedRoles={['operator', 'analyst', 'admin']} />}>
        <Route path="/dashboard" element={<DashboardPage />} />
      </Route>

      {/* Default redirect */}
      <Route
        path="/"
        element={
          operator
            ? <Navigate to="/dashboard" replace />
            : <Navigate to="/login" replace />
        }
      />

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
