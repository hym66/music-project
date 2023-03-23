import Vue from 'vue'
import App from './App.vue'
import { upload,alert } from 'element-ui'
import http from 'axios'
import store from './store/index'
Vue.config.productionTip = false
Vue.use(upload)
Vue.use(alert)

import {
  Message,
} from 'element-ui'


new Vue({
  store,//Vuex注册
  render: h => h(App),
}).$mount('#app')
//全局baseURL
//http.defaults.baseURL = "http://127.0.0.1:8000"
//注册axios
Vue.prototype.$http = http
Vue.prototype.$message = Message;


// import Axios from 'axios'
// Axios.defaults.baseURL = '/'
