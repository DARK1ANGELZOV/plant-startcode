'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import Image from 'next/image'
import { useAuth } from '@/contexts/AuthContext'
import toast from 'react-hot-toast'
import { Eye, EyeOff, Loader2, Check, X } from 'lucide-react'

export default function RegisterPage() {
  const router = useRouter()
  const { register, continueAsGuest } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const passwordRequirements = [
    { label: 'Минимум 8 символов', met: password.length >= 8 },
    { label: 'Есть цифра', met: /\d/.test(password) },
    { label: 'Есть буква', met: /[a-zA-Zа-яА-Я]/.test(password) },
  ]

  const isPasswordValid = passwordRequirements.every((req) => req.met)
  const doPasswordsMatch = password === confirmPassword && confirmPassword.length > 0

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!email || !password || !confirmPassword) {
      toast.error('Заполните все поля')
      return
    }

    if (!isPasswordValid) {
      toast.error('Пароль не соответствует требованиям')
      return
    }

    if (!doPasswordsMatch) {
      toast.error('Пароли не совпадают')
      return
    }

    setIsLoading(true)
    try {
      await register(email, password)
      toast.success('Регистрация успешна')
      router.push('/')
    } catch {
      toast.error('Ошибка регистрации')
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
          <h1 className="text-xl font-semibold text-white mb-6 text-center">Регистрация</h1>

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
                  placeholder="Придумайте пароль"
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

              {password.length > 0 && (
                <div className="mt-2 space-y-1">
                  {passwordRequirements.map((req, index) => (
                    <div key={index} className="flex items-center gap-2 text-xs">
                      {req.met ? <Check size={14} className="text-emerald-500" /> : <X size={14} className="text-red-500" />}
                      <span className={req.met ? 'text-emerald-500' : 'text-red-500'}>{req.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block text-sm text-white/70 mb-2">
                Подтвердите пароль
              </label>
              <div className="relative">
                <input
                  id="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className={`w-full px-4 py-3 pr-12 bg-white/5 border rounded-lg text-white placeholder:text-white/40 focus:outline-none transition-all ${
                    confirmPassword.length > 0
                      ? doPasswordsMatch
                        ? 'border-emerald-500 focus:ring-1 focus:ring-emerald-500'
                        : 'border-red-500 focus:ring-1 focus:ring-red-500'
                      : 'border-white/10 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500'
                  }`}
                  placeholder="Повторите пароль"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-white/50 hover:text-white/80 transition-colors focus:outline-none"
                  tabIndex={-1}
                >
                  {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
              {confirmPassword.length > 0 && !doPasswordsMatch && <p className="mt-1 text-xs text-red-500">Пароли не совпадают</p>}
            </div>

            <button
              type="submit"
              disabled={isLoading || !isPasswordValid || !doPasswordsMatch}
              className="w-full py-3 bg-emerald-500 hover:bg-emerald-600 text-white font-medium rounded-lg transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  Регистрация...
                </>
              ) : (
                'Зарегистрироваться'
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-white/60 text-sm">
              Уже есть аккаунт?{' '}
              <Link href="/login" className="text-emerald-500 hover:text-emerald-400 transition-colors">
                Войти
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
