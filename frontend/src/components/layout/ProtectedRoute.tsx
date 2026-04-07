import { Navigate, Outlet } from 'react-router-dom'
import { useUIStore } from '@/store/ui.store'
import type { UserRole } from '@/types'

interface ProtectedRouteProps {
  allowedRoles: UserRole[]
}

export function ProtectedRoute({ allowedRoles }: ProtectedRouteProps) {
  const operator = useUIStore((s) => s.operator)

  // No operator session → go to login
  if (!operator) {
    return <Navigate to="/login" replace />
  }

  // Wrong role → go to unauthorised
  if (!allowedRoles.includes(operator.role)) {
    return <Navigate to="/unauthorised" replace />
  }

  return <Outlet />
}
