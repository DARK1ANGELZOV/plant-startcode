'use client'

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import type { User } from '@/lib/types'

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  isGuest: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  logout: () => void
  continueAsGuest: () => void
  checkGuestTrialUsed: () => boolean
  setGuestTrialUsed: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isGuest, setIsGuest] = useState(false)

  // Check auth status on mount
  useEffect(() => {
    const checkAuth = async () => {
      if (api.isAuthenticated()) {
        try {
          const userData = await api.getMe()
          setUser(userData)
        } catch {
          api.clearAuth()
        }
      } else {
        // Check if user was previously a guest
        const wasGuest = localStorage.getItem('is_guest') === 'true'
        if (wasGuest) {
          setIsGuest(true)
        }
      }
      setIsLoading(false)
    }

    checkAuth()
  }, [])

  const login = async (email: string, password: string) => {
    const response = await api.login({ email, password })
    api.setAuth(response.access_token, response.user.id)
    setUser(response.user)
    setIsGuest(false)
    localStorage.removeItem('is_guest')
  }

  const register = async (email: string, password: string) => {
    await api.register({ email, password })
    // Auto-login after registration
    await login(email, password)
  }

  const logout = useCallback(() => {
    api.clearAuth()
    setUser(null)
    setIsGuest(false)
    localStorage.removeItem('is_guest')
  }, [])

  const continueAsGuest = () => {
    setIsGuest(true)
    localStorage.setItem('is_guest', 'true')
    // Initialize guest trial flag if not exists
    if (localStorage.getItem('guestTrialUsed') === null) {
      localStorage.setItem('guestTrialUsed', 'false')
    }
  }

  const checkGuestTrialUsed = (): boolean => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem('guestTrialUsed') === 'true'
  }

  const setGuestTrialUsed = () => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('guestTrialUsed', 'true')
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        isGuest,
        login,
        register,
        logout,
        continueAsGuest,
        checkGuestTrialUsed,
        setGuestTrialUsed,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
