/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Space Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        surface: { DEFAULT: "#141820", border: "#252a35" },
      },
    },
  },
  plugins: [],
};
