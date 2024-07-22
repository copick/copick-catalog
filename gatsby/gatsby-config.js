module.exports = {
  pathPrefix: `/`,
  siteMetadata: {
    title: 'copick album catalog',
    subtitle: 'sharing copick tools',
    catalog_url: 'https://github.com/copick/copick-catalog',
    menuLinks:[
      {
         name:'Catalog',
         link:'/catalog'
      },
      {
         name:'About',
         link:'/about'
      },
    ]
  },
  plugins: [{ resolve: `gatsby-theme-album`, options: {} }],
}
