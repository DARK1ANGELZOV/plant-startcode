'use client'

import Link from 'next/link'
import Image from 'next/image'
import { useAuth } from '@/contexts/AuthContext'
import { Menu, User, LogOut, ChevronDown } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'

interface NavbarProps {
  onMenuClick: () => void
}

export function Navbar({ onMenuClick }: NavbarProps) {
  const { user, isAuthenticated, logout } = useAuth()
  const [showDropdown, setShowDropdown] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <header className="fixed top-0 left-0 right-0 h-14 bg-[#0f0f0f] border-b border-white/10 z-50 flex items-center justify-between px-4">
      <div className="flex items-center gap-3">
        <button onClick={onMenuClick} className="md:hidden p-2 hover:bg-white/10 rounded-lg transition-colors" aria-label="Открыть меню">
          <Menu size={20} className="text-white" />
        </button>

        <Link href="/" className="flex items-center gap-2 group h-11 md:h-11">
          <Image src="/logo.png" alt="PlantVision AI" width={44} height={44} className="w-9 h-9 md:w-11 md:h-11 transition-transform duration-200" />
          <span className="text-lg md:text-xl font-semibold text-[#10b981] hidden sm:inline transition-all duration-200 group-hover:text-[#34d399] group-hover:scale-105 origin-left">
            PlantVision AI
          </span>
        </Link>
      </div>

      <div className="flex items-center gap-2">
        {isAuthenticated ? (
          <div className="relative" ref={dropdownRef}>
            <button onClick={() => setShowDropdown(!showDropdown)} className="flex items-center gap-2 px-3 py-2 hover:bg-white/10 rounded-lg transition-colors">
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <User size={18} className="text-emerald-500" />
              </div>
              <span className="text-white/80 text-sm hidden sm:inline max-w-[120px] truncate">{user?.email}</span>
              <ChevronDown size={16} className="text-white/60" />
            </button>

            {showDropdown && (
              <div className="absolute right-0 top-full mt-2 w-48 glass rounded-lg py-1 animate-fade-in">
                <Link
                  href="/profile"
                  className="flex items-center gap-3 px-4 py-2 text-white/80 hover:bg-white/10 transition-colors"
                  onClick={() => setShowDropdown(false)}
                >
                  <User size={18} />
                  <span>Профиль</span>
                </Link>
                <button
                  onClick={() => {
                    logout()
                    setShowDropdown(false)
                  }}
                  className="w-full flex items-center gap-3 px-4 py-2 text-white/80 hover:bg-white/10 transition-colors"
                >
                  <LogOut size={18} />
                  <span>Выйти</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          <Link
            href="/login"
            className="px-4 py-2 text-emerald-500 border border-emerald-500/50 hover:bg-emerald-500/10 rounded-lg transition-all duration-200 text-sm font-medium"
          >
            Войти
          </Link>
        )}
      </div>
    </header>
  )
}
