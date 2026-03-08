import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { Toaster } from 'react-hot-toast'
import { AuthProvider } from '@/contexts/AuthContext'
import './globals.css'

const inter = Inter({
  subsets: ['latin', 'cyrillic'],
  variable: '--font-inter',
  weight: ['400', '500', '600', '700'],
})

export const metadata: Metadata = {
  title: 'PlantVision AI - Анализ растений',
  description: 'Платформа для AI-анализа растений: чат, сегментация, морфометрия и рекомендации.',
  generator: 'PlantVision AI',
  icons: {
    icon: '/logo.png',
    apple: '/logo.png',
  },
}

export const viewport: Viewport = {
  themeColor: '#0f0f0f',
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="ru" className="dark" style={{ backgroundColor: '#0f0f0f' }}>
      <head>
        <style
          dangerouslySetInnerHTML={{
            __html: `
          html, body { background-color: #0f0f0f !important; }
        `,
          }}
        />
      </head>
      <body className={`${inter.variable} font-sans antialiased bg-[#0f0f0f] text-white font-medium`} style={{ backgroundColor: '#0f0f0f' }}>
        <AuthProvider>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              style: {
                background: 'rgba(20, 20, 20, 0.95)',
                color: '#fff',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                backdropFilter: 'blur(10px)',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#fff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </AuthProvider>
        <Analytics />
      </body>
    </html>
  )
}
