import { Routes, Route } from 'react-router-dom'
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';


import { Upload } from './components/upload'
import { Home, PageNotFound } from './pages';



function App() {


  return (
    <>

      <ToastContainer />
      <Routes>
        <Route path='/' element={<Home />}></Route>
        <Route path='/upload' element={<Upload />} />
        <Route path="*" element={<PageNotFound />} />
      </Routes>


    </>
  )
}

export default App
