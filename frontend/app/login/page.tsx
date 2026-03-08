'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import Image from 'next/image'
import { useAuth } from '@/contexts/AuthContext'
import toast from 'react-hot-toast'
import { Eye, EyeOff, Loader2 } from 'lucide-react'

export default function LoginPage() {
  const router = useRouter()
  const { login, continueAsGuest } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!email || !password) {
      toast.error('Заполните email и пароль')
      return
    }

    setIsLoading(true)
    try {
      await login(email, password)
      toast.success('Вход выполнен')
      router.push('/')
    } catch {
      toast.error('Неверный email или пароль')
    } finally {
      setIsLoading(false)
    }
  }

  const handleGuestContinue = () => {
    continueAsGuest()
    router.push('/')
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0f0f0f] p-4">
      <div className="w-full max-w-md animate-fade-in">
        <div className="flex flex-col items-center mb-8">
          <Link href="/" className="flex items-center gap-3 mb-2 group">
            <Image src="/logo.png" alt="PlantVision AI" width={56} height={56} className="w-14 h-14" />
            <span className="text-2xl font-semibold text-[#10b981] transition-all duration-200 group-hover:text-[#34d399] group-hover:scale-105 origin-left">
              PlantVision AI
            </span>
          </Link>
          <p className="text-white/70 text-sm">AI-анализ растений</p>
        </div>

        <div className="glass-card">
          <h1 className="text-xl font-semibold text-white mb-6 text-center">Вход</h1>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="email" className="block text-sm text-white/70 mb-2">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/40 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all"
                placeholder="example@email.com"
                disabled={isLoading}
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm text-white/70 mb-2">
                Пароль
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 pr-12 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/40 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all"
                  placeholder="Введите пароль"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-white/50 hover:text-white/80 transition-colors focus:outline-none"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 bg-emerald-500 hover:bg-emerald-600 text-white font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  Вход...
                </>
              ) : (
                'Войти'
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-white/60 text-sm">
              Нет аккаунта?{' '}
              <Link href="/register" className="text-emerald-500 hover:text-emerald-400 transition-colors">
                Зарегистрироваться
              </Link>
            </p>
          </div>

          <div className="flex items-center gap-4 my-6">
            <div className="flex-1 h-px bg-white/10" />
            <span className="text-white/40 text-sm">или</span>
            <div className="flex-1 h-px bg-white/10" />
          </div>

          <button
            onClick={handleGuestContinue}
            className="w-full py-3 border border-white/20 hover:border-white/40 text-white/80 hover:text-white font-medium rounded-lg transition-all duration-200"
          >
            Продолжить как гость
          </button>
        </div>
      </div>
    </div>
  )
}
