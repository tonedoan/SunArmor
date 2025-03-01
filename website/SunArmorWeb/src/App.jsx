import { useState } from 'react'
import './App.css'
import ImageLoad from './components/ImageLoad'
import logo from './assets/sunArmorLogo1.png'

function App() {
    const [count, setCount] = useState(0)

    return (
        <>
            <div className='py-2 flex flex-row items-center space-x-4 justify-center'>
                <span className='text-5xl font-bold'>SunArmor AI</span>
                <img src={logo} alt='SunArmor AI' className='w-20 h-20' />
            </div>

            <ImageLoad />
        </>
    )
}

export default App
