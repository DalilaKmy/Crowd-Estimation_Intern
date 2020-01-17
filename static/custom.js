// Initialise Pusher

const pusher = new Pusher('key', {
    cluster: 'ap1',
    encrypted: true
});

var channel = pusher.subscribe('table');

channel.bind('new-record', (data) => {

   $('#namesAll').append(`
        <tr id="${data.data.id}">
            <th scope="row"> ${data.data.peopleIn} </th>
            <td> ${data.data.peopleOut} </td>
            <td> ${data.data.totalPeople} </td>
            <td> ${data.data.dateTime} </td>
        </tr>

   `)
});

channel.bind('update-record', (data) => {


    $(`#${data.data.id=1}`).html(`
        <th scope="row"> ${data.data.peopleIn} </th>
        <td> ${data.data.peopleOut} </td>
        <td> ${data.data.totalPeople} </td>
        <td> ${data.data.dateTime} </td>
    `)

 });