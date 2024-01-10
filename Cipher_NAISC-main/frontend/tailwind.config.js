/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Share Tech Mono"', 'monospace'],
        body: ['"Barlow"', 'sans-serif'],
        cond: ['"Barlow Condensed"', 'sans-serif'],
      },
      colors: {
        // Base backgrounds
        bg: {
          0: '#04080c',
          1: '#080f16',
          2: '#0c1520',
          3: '#101c28',
          4: '#152232',
        },
        // Borders
        border: {
          DEFAULT: '#1a2d3f',
          2: '#1f3550',
          3: '#254060',
        },
        // Text
        text: {
          1: '#ddeeff',
          2: '#7aa8cc',
          3: '#3d6685',
          4: '#1f3d55',
        },
        // Severity
        critical: '#ff2244',
        high: '#ff8c00',
        medium: '#ffd600',
        low: '#00e5aa',
        info: '#00aaff',
        cloud: '#00e5aa',
        degraded: '#ff8c00',
      },
      animation: {
        'blink': 'blink 0.9s step-end infinite',
        'ticker': 'ticker 30s linear infinite',
        'pulse-zone': 'pulseZone 1.5s ease-in-out infinite',
        'slide-in': 'slideIn 0.5s ease',
        'inc-pulse': 'incPulse 2s ease-in-out infinite',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        ticker: {
          '0%': { transform: 'translateX(100%)' },
          '100%': { transform: 'translateX(-100%)' },
        },
        pulseZone: {
          '0%, 100%': { backgroundColor: 'rgba(255,34,68,0.10)' },
          '50%': { backgroundColor: 'rgba(255,34,68,0.18)' },
        },
        slideIn: {
          '0%': { background: 'rgba(0,229,170,0.07)', transform: 'translateX(-4px)' },
          '100%': { background: 'transparent', transform: 'translateX(0)' },
        },
        incPulse: {
          '0%, 100%': { backgroundColor: 'rgba(255,34,68,0.07)' },
          '50%': { backgroundColor: 'rgba(255,34,68,0.13)' },
        },
      },
    },
  },
  plugins: [],
}
